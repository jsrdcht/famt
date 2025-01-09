from functools import partial

import torch
import torch.nn.functional as F
import torch.nn as nn
# from timm.models.vision_transformer import Block, PatchEmbed
from timm.models.vision_transformer import PatchEmbed

from util.pos_embed import get_2d_sincos_pos_embed

from timm.models.vision_transformer import Block as TimmBlock
from timm.models.vision_transformer import Attention as TimmAttention
from timm.models.vision_transformer import DropPath

class Attention(TimmAttention):
    """
    继承自timm.models.vision_transformer.Attention, 
    并在forward方法中添加return_attention参数控制输出注意力图。
    """
    def forward(self, x, return_attention=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)


        # 兼容fused与非fused的两种分支
        if self.fused_attn and not return_attention:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p,
            )
            # 注意：fused版本的torch会自动计算并返回注意力，所以若需return_attention
            # 可考虑进一步修改torch的scaled_dot_product_attention源码
            # 这里只是给出一个示例，如果你确实需要，也可以手动写非fused逻辑
            attn = None
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention and attn is not None:
            return x, attn
        else:
            return x, None  # 若fused或者不需要attention，返回None

class Block(TimmBlock):
    """
    继承自timm.models.vision_transformer.Block, 
    并在forward方法中支持return_attention控制是否返回注意力图。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 将attention替换为自定义的MyAttention
        # 注意：在timm的Block里，self.attn是在__init__里创建的
        # 这里需要手动替换成MyAttention
        self.attn = Attention(
            dim=self.attn.qkv.in_features,
            num_heads=self.attn.num_heads,
            qkv_bias=(self.attn.qkv.bias is not None),
            qk_norm=False,  # 如果你的timm版本里是True，也要保持一致
            attn_drop=self.attn.attn_drop.p,
            proj_drop=self.attn.proj_drop.p,
            norm_layer=self.attn.q_norm.__class__,
        )

    def forward(self, x, return_attention=False):
        """
        在forward中增加对return_attention的支持。
        """
        # 第一部分：Multi-Head Self Attention
        if not return_attention:
            # 如果不需要attention，跟原先一样
            x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))[0]))
            # [0] 是因为 MyAttention.forward 返回 (x, attn)
        else:
            # 如果需要attention，就获取 attn
            attn_input = self.norm1(x)
            out, attn_map = self.attn(attn_input, return_attention=True)
            x = x + self.drop_path1(self.ls1(out))

        # 第二部分：MLP
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        if return_attention:
            return x, attn_map
        else:
            return x


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def patchify_map(self, map):
        """
        map: (B, 1, H, W)
        x: (B, patch_size**2)
        """
        p = self.patch_embed.patch_size[0]

 
        h = w = map.shape[2] // p
        map = map.squeeze(1)
        x = map.reshape(shape=(map.shape[0], h, p, w, p))
        x = torch.einsum('bhpwq->bhwpq', x)
        x = x.reshape(shape=(map.shape[0], h * w, p**2))
        return x 

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs
    
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore



    def amt_masking_throwing(self, x, mask_ratio, throw_ratio, mask_weights):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
    
        N, L, D = x.shape  # batch, length, dim


        len_mask_tail = int(L * mask_ratio)
        len_keep_head = int(L * (mask_ratio + throw_ratio))

        mask_weights = self.patchify_map(mask_weights)
        mask_weights = mask_weights.sum(-1)

        #sampling from masking weights
        ids_shuffle = torch.multinomial(mask_weights, L) 
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, len_keep_head:]

        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        

        # generate the binary mask: 0 is keep, 1 is masked, -1 is thrown
        mask = torch.ones([N, L], device=x.device)
        mask[:, len_keep_head:] = 0
        mask[: , len_mask_tail:len_keep_head] = -1

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore) 

        return x_masked, mask, ids_restore

    def get_pos_embed(self):
        return self.pos_embed[:,1:,:]

    def forward_encoder(self, x, mask_ratio, throw_ratio, mask_weights, epoch):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        
        if epoch > 40:#start epoch of amt
            x, mask, ids_restore = self.amt_masking_throwing(x, mask_ratio, throw_ratio, mask_weights)
        else :
            x, mask, ids_restore = self.random_masking(x, mask_ratio=0.75)

        
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)


        # apply Transformer blocks
        for i,blk in enumerate(self.blocks):
            x = blk(x)
       
        x = self.norm(x)
        
        
        return x, mask, ids_restore


    def forward_encoder_test(self,x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
       
        # apply Transformer blocks
        for i,blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                return blk(x, return_attention=True)
 
        
        return x





    def forward_decoder(self, x, ids_restore, mask_flag, mask_ratio, epoch):


        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        

        if epoch > 40:
            mask_tokens = self.mask_token.repeat(x.shape[0], int(ids_restore.shape[1] * mask_ratio), 1)
            throw_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1] - int(ids_restore.shape[1] * mask_ratio), 1)

            x_ = torch.cat([mask_tokens, throw_tokens, x[:, 1:, :]], dim=1)  # no cls token
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  

        else:
            mask_tokens = self.mask_token.repeat(x.shape[0], int(ids_restore.shape[1] * 0.75), 1)
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        #delete thrown tokens, save computing costs
        if epoch > 40:
            x_d = x[:, 1:, :][(mask_flag+1).bool(), :].reshape(x.shape[0],-1,x.shape[2])
            x = torch.cat([x[:, :1, :], x_d], dim=1)


        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask_flag, epoch):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        #delete useless target patches
        if epoch > 40:
            target = target[(mask_flag+1).bool(), :].reshape(target.shape[0],-1,target.shape[2])
            mask_new = mask_flag[mask_flag != -1].reshape(mask_flag.shape[0],-1)
        else:
            mask_new = mask_flag


        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask_new).sum() / mask_new.sum()  # mean loss only on masked patches 

        return loss
 
    def forward(self, imgs, mask_weights, mask_ratio=0.45, throw_ratio=0.4, epoch = 0):

        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio, throw_ratio, mask_weights, epoch)
        pred = self.forward_decoder(latent, ids_restore, mask, mask_ratio, epoch)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask, epoch)

        return loss, pred, mask

def mae_vit_small_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=128, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_small_patch16 = mae_vit_small_patch16_dec512d8b
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
model = MaskedAutoencoderViT()
model.forward_encoder
