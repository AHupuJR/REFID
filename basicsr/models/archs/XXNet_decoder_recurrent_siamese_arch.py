import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn import init
from basicsr.models.archs.recurrent_sub_modules import ConvLayer, UpsampleConvLayer, TransposedConvLayer, \
    RecurrentConvLayer, ResidualBlock, ConvLSTM, ConvGRU, ImageEncoderConvBlock, SimpleRecurrentConvLayer, SimpleRecurrentThenDownConvLayer, \
        TransposeRecurrentConvLayer
from basicsr.models.archs.dcn_util import ModulatedDeformConvPack
from basicsr.models.archs.fusion_modules import img_ev_fusion, AttenPred
from einops import rearrange


def skip_concat(x1, x2):
    return torch.cat([x1, x2], dim=1)


def skip_sum(x1, x2):
    return x1 + x2


class DecoderRecurrentAttenResidualUNet(nn.Module):
    def __init__(self, img_chn, ev_chn, out_chn=3, skip_type='sum', activation='sigmoid',
                 num_encoders=3, base_num_channels=32, num_residual_blocks=2, norm=None, use_recurrent_upsample_conv=True):
        super(DecoderRecurrentAttenResidualUNet, self).__init__()

        self.ev_chn = ev_chn
        self.img_chn = img_chn
        self.out_chn = out_chn
        self.skip_type = skip_type
        self.apply_skip_connection = skip_sum if self.skip_type == 'sum' else skip_concat
        self.activation = activation
        self.norm = norm

        if use_recurrent_upsample_conv:
            print('Using Recurrent UpsampleConvLayer (slow, but recurrent in decoder)')
            self.UpsampleLayer = TransposeRecurrentConvLayer
        else:
            print('Using No recurrent UpsampleConvLayer (fast, but no recurrent in decoder)')
            self.UpsampleLayer = UpsampleConvLayer

        self.num_encoders = num_encoders
        self.base_num_channels = base_num_channels
        self.num_residual_blocks = num_residual_blocks
        self.max_num_channels = self.base_num_channels * pow(2, self.num_encoders)

        assert(self.ev_chn > 0)
        assert(self.img_chn > 0)
        assert(self.out_chn > 0)

        self.encoder_input_sizes = []
        for i in range(self.num_encoders):
            self.encoder_input_sizes.append(self.base_num_channels * pow(2, i))

        self.encoder_output_sizes = [self.base_num_channels * pow(2, i + 1) for i in range(self.num_encoders)]

        self.activation = getattr(torch, self.activation, 'sigmoid')

    def build_resblocks(self):
        self.resblocks = nn.ModuleList()
        for i in range(self.num_residual_blocks):
            self.resblocks.append(ResidualBlock(self.max_num_channels, self.max_num_channels, norm=self.norm))

    def build_decoders(self):
        decoder_input_sizes = list(reversed([self.base_num_channels * pow(2, i + 1) for i in range(self.num_encoders)]))

        self.decoders = nn.ModuleList()
        for input_size in decoder_input_sizes:
            self.decoders.append(self.UpsampleLayer(input_size if self.skip_type == 'sum' else 2 * input_size,
                                                    input_size // 2,
                                                    kernel_size=2, padding=0, norm=self.norm)) # kernei_size= 5, padidng =2 before
    ## Atten Pred
    def build_prediction_layer(self):
        self.pred = ConvLayer(self.base_num_channels if self.skip_type == 'sum' else 2 * self.base_num_channels,
                              self.out_chn, kernel_size=3, stride=1, padding=1, relu_slope=None, norm=self.norm)
        # self.pred = AttenPred(self.base_num_channels)



class UNetDecoderRecurrentSiameseImg(DecoderRecurrentAttenResidualUNet):
    """
    Recurrent UNet architecture where every encoder is followed by a recurrent convolutional block,
    such as a ConvLSTM or a ConvGRU.
    Symmetric, skip connections on every encoding layer.

    num_block: the number of blocks in each simpleconvlayer.
    """

    def __init__(self, img_chn, ev_chn, out_chn=3, skip_type='sum',
                 recurrent_block_type='convlstm', activation='sigmoid', num_encoders=4, base_num_channels=32,
                 num_residual_blocks=2, norm=None, use_recurrent_upsample_conv=True, num_block=3, use_first_dcn=False):
        super(UNetDecoderRecurrentSiameseImg, self).__init__(img_chn, ev_chn, out_chn, skip_type, activation,
                                            num_encoders, base_num_channels, num_residual_blocks, norm,
                                            use_recurrent_upsample_conv)

        ## event
        self.head = ConvLayer(self.ev_chn, self.base_num_channels,
                              kernel_size=5, stride=1, padding=2, relu_slope=0.2)  # N x C x H x W -> N x 32 x H x W
        self.encoders = nn.ModuleList()
        self.img_ev_fusions = nn.ModuleList()

        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            ## fusion
            self.img_ev_fusions.append(
                img_ev_fusion(num_channels=output_size)
            )
            ## encoders
            if recurrent_block_type == 'simpleconv':
                self.encoders.append(SimpleRecurrentConvLayer(input_size, output_size,
                                                        kernel_size=5, stride=2, padding=2,
                                                        norm=self.norm, num_block=num_block, use_first_dcn=use_first_dcn))
            elif recurrent_block_type == 'simpleconvThendown':
                self.encoders.append(SimpleRecurrentThenDownConvLayer(input_size, output_size,
                                                        kernel_size=3, stride=1, padding=1, # the param of the first conv
                                                        norm=self.norm, num_block=num_block, use_first_dcn=use_first_dcn))
            else:
                self.encoders.append(RecurrentConvLayer(input_size, output_size,
                                                        kernel_size=5, stride=2, padding=2,
                                                        recurrent_block_type=recurrent_block_type,
                                                        norm=self.norm))
        ## img
        self.head_img = ConvLayer(self.img_chn, self.base_num_channels,
                              kernel_size=5, stride=1, padding=2, relu_slope=0.2)  # N x C x H x W -> N x 32 x H x W
        self.img_encoders = nn.ModuleList()
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            self.img_encoders.append(ImageEncoderConvBlock(in_size=input_size, out_size=output_size,
                                                            downsample=True, relu_slope=0.2))
        

        self.build_resblocks()
        self.build_decoders()
        self.build_prediction_layer()


    def forward(self, x, event):
        """
        :param x: b 2 c h w -> b, 2c, h, w
        :param event: b, t, num_bins, h, w -> b*t num_bins(2) h w 
        :return: b, t, out_chn, h, w

        One direction propt version
        TODO: add bi-direction propt version
        """
        # reshape
        if x.dim()==5:
            x = rearrange(x, 'b t c h w -> b (t c) h w')
        b, t, num_bins, h, w = event.size()
        event = rearrange(event, 'b t c h w -> (b t) c h w')

        x0, x1 = torch.chunk(x, chunks=2, dim=1)
        img0 = x0[:,:3]
        img1 = x1[:,:3]
        imgs = torch.cat((img0, img1), dim=1)

        # siamese
        x = rearrange(x, 'b (t c) h w -> (t b) c h w', t=2)

        # head
        x = self.head_img(x) # image feat
        # head = x
        e = self.head(event)   # event feat
        # image encoder
        x_blocks = []
        for i, img_encoder in enumerate(self.img_encoders):
            x = img_encoder(x)
            ## siamese
            x0_x1_feat = torch.chunk(x, chunks=2, dim=0)
            x_blocks.append(x0_x1_feat) # tuple of feat of x0 and x1

        
        ## forward propt 
        e = rearrange(e, '(b t) c h w -> b t c h w', b=b, t=t)
        out_l = []
        prev_states = [None] * self.num_encoders
        prev_states_decoder = [None] * self.num_encoders

        for frame_idx in range(0,t):
            e_blocks = [] # initial skip feats for each frame
            e_cur = e[:, frame_idx,:,:,:] # b,c,h,w
            ### event encoder
            for i, encoder in enumerate(self.encoders):
                e_cur, state = encoder(e_cur, prev_states[i])
                ## fusion
                e_cur = self.img_ev_fusions[i](e_cur, x_blocks[i][0], x_blocks[i][1])
                e_blocks.append(e_cur)
                prev_states[i] = state # update state for next frame
            # residual blocks
            for resblock in self.resblocks:
                e_cur = resblock(e_cur)

            ## Decoder
            for i, decoder in enumerate(self.decoders):
                e_cur, state = decoder(self.apply_skip_connection(e_cur, e_blocks[self.num_encoders - i - 1]), prev_states_decoder[i])
                prev_states_decoder[i] = state

            # attention tail
            # out = self.pred(e_cur, imgs)

            # ordinary tail
            out = self.pred(e_cur)

            out_l.append(out)
        
        return torch.stack(out_l, dim=1) # b,t,c,h,w

