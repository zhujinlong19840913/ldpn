import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
import paddle
from paddleseg import utils
from paddleseg.cvlibs import manager, param_init
from paddleseg.models import layers

@manager.MODELS.add_component
class DPN(nn.Layer):
    def __init__(self,
                 num_classes,                
                 channels=None,
                 pretrained=None):
        super(DPN, self).__init__()

        #68
        init_num_filter = 64
        # conv1
        self.conv1 = layers.ConvBNReLU(
            in_channels=3,
            out_channels=init_num_filter,
            kernel_size=3,
            padding=1,
            stride=2)

        self.pool1 = nn.MaxPool2D(kernel_size=3,stride=2,padding=1)

        self.conv_list = []
        self.down_sample = Encoder(init_num_filter)
        
        self.decode = Decoder(False)
        self.us1 = UpSampling(32, 64, 360,False)
        
        #68
        channeldpn=64

        self.cls256 = nn.Conv2D(
            in_channels=init_num_filter,
            out_channels=num_classes,
            kernel_size=3,
            stride=1,
            padding=1)

        self.cls128 = nn.Conv2D(
            in_channels=360,
            out_channels=num_classes,
            kernel_size=3,
            stride=1,
            padding=1)

        self.cls64 = nn.Conv2D(
            in_channels=400,
            out_channels=num_classes,
            kernel_size=3,
            stride=1,
            padding=1)

        self.cls32 = nn.Conv2D(
            in_channels=420,
            out_channels=num_classes,
            kernel_size=3,
            stride=1,
            padding=1)

        self.cls = nn.Conv2D(
            in_channels=channeldpn,
            out_channels=num_classes,
            kernel_size=3,
            stride=1,
            padding=1)
        self.pretrained = None

    def forward(self, x):
        conv = self.conv1(x)
        out = self.pool1(conv)

        out,short_cuts_conv = self.down_sample(out)
        
        out = self.decode(out,short_cuts_conv)
        logit_list = []

        out = self.us1(out, conv)
        out = F.upsample(x=out, size=[512,512],mode = 'bilinear')
        
        logit = self.cls(out)
        logit256 = self.cls256(conv)
        logit128 = self.cls128(short_cuts_conv[0])
        logit64 = self.cls64(short_cuts_conv[1])
        logit32 = self.cls32(short_cuts_conv[2])

        logit_list.append(logit)
        logit_list.append(logit256)
        logit_list.append(logit128)
        logit_list.append(logit64)
        logit_list.append(logit32)

        return logit_list

class Dual_path_factory(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_1x1_a,
                 num_3x3_b,
                 num_1x1_c,
                 inc,
                 G,
                 _type='normal',layer_idx=1):
        super(Dual_path_factory, self).__init__()

        kw = 3
        kh = 3
        pw = (kw - 1) // 2
        ph = (kh - 1) // 2
        
        self.key_stride = 1

        if _type is 'proj':
            key_stride = 1
            has_proj = True
        if _type is 'down':
            key_stride = 2
            self.key_stride = 2
            has_proj = True
        if _type is 'normal':
            key_stride = 1
            has_proj = False

        self.has_proj = has_proj
        self.num_1x1_c = num_1x1_c
        self.inc = inc

        if self.has_proj:
            self.c1x1_w = layers.ConvBNReLU(
                out_channels=(num_1x1_c + 2 * inc),
                kernel_size=(1, 1),
                in_channels=num_channels,
                padding=(0, 0),
                stride=(key_stride, key_stride)
            )

        self.c1x1_a = layers.ConvBNReLU(
            out_channels=num_1x1_a,
            kernel_size=(1, 1),
            in_channels=num_channels,
            padding=(0, 0)
        )

        self.c3x3_b = layers.ConvBNReLU(
            out_channels=num_3x3_b,
            kernel_size=(kw, kh),
            in_channels=num_1x1_a,
            padding=(pw, ph),
            stride=(key_stride, key_stride),
            cgroups=G
        )

        self.c1x1_c = layers.ConvBNReLU(
            out_channels=(num_1x1_c + inc),
            kernel_size=(1, 1),
            in_channels=num_3x3_b,
            padding=(0, 0)
        )

    def forward(self, data):
        # PROJ
        if type(data) is list:
            data_in = paddle.concat([data[0], data[1]], axis=1)
        else:
            data_in = data

        if self.has_proj:
            x = self.c1x1_w(data_in)
            # x = self.c1x1_w_att(x)
            data_o1, data_o2 = paddle.fluid.layers.split(
                x,
                num_or_sections=[self.num_1x1_c, 2 * self.inc],
                dim=1
            )
        else:
            data_o1 = data[0]
            data_o2 = data[1]

        out = self.c1x1_a(data_in)
        out = self.c3x3_b(out)
        out = self.c1x1_c(out)

        c1x1_c1, c1x1_c2 = paddle.fluid.layers.split(
            out,
            num_or_sections=[self.num_1x1_c, self.inc],
            dim=1
        )

        # OUTPUTS
        summ = paddle.fluid.layers.elementwise_add(
            x=data_o1, y=c1x1_c1)
        dense = paddle.concat(
            [data_o2, c1x1_c2], axis=1)

        return [summ, dense]

class Encoder(nn.Layer):
    def __init__(self,channels1):
        super().__init__()
        k_R = 128
        G   = 32
        k_sec  = {  2: 3,  3: 4,  4: 12, 5: 3   }
        inc_sec= {  2: 16, 3: 32, 4: 32, 5: 64  }
        bws =    {  2: 64, 3: 128,4: 256,5: 512 }
        conv_channels = ([ i==1 and channels1 or (bws[i]+inc_sec[i]*3+(k_sec[i]-1)*inc_sec[i]) for i in range(1,6)])
        channeloutput = [420, 400, 360, 64]

        self.conv_sample_list = nn.LayerList([ i == len(conv_channels)-1 and nn.Sequential(layers.ConvBNReLU(conv_channels[i],conv_channels[i],3),\
            layers.ConvBNReLU(conv_channels[i],1000,3)) \
            or nn.Sequential(layers.ConvBNReLU(conv_channels[i],conv_channels[i],3), \
            layers.ConvBNReLU(conv_channels[i],channeloutput[len(channeloutput)-1-i],3)) \
            for i in range(len(conv_channels))])


        downSequential = []
        for i in range(2,6): 
            conv_list = []
            bw = bws[i]
            inc = inc_sec[i]
            R = (k_R*bw)//64
            if i == 2:
                _type1 = 'proj'
                _type2 = 'normal'
            else:
                _type1 = 'down'
                _type2 = 'normal'
            conv2_x_x = Dual_path_factory(channels1, R, R, bw, inc, G, _type1,i-1)
            conv_list.append(conv2_x_x)
            channels1 = bw + inc + 2 * inc
            for i_ly in range(2,k_sec[i]+1):
                conv2_x_x = Dual_path_factory(channels1, R, R, bw, inc, G, _type2,i-1)
                conv_list.append(conv2_x_x)
                channels1 = channels1 + inc
            # print("i:"+str(i))

            downSequential.append(nn.Sequential(*conv_list))
        
        self.down_sample_list = nn.LayerList([sequentialItem for sequentialItem in downSequential])

    def forward(self, x):
        short_conv_cuts = []
        index = 0
        for down_sample in self.down_sample_list:
            x = down_sample(x)
            x = paddle.fluid.layers.concat(x, axis=1)

            if index<3:  
                y_conv = self.conv_sample_list[index+1](x)
                short_conv_cuts.append(y_conv)
                index=index+1

        x = self.conv_sample_list[index+1](x)
        return x,short_conv_cuts

class Decoder(nn.Layer):
    def __init__(self, align_corners, use_deconv=False):
        super().__init__()
        up_channels = [[210, 420,1000], [200, 400,420], [180, 360,400], [32, 128,360]]
        
        self.up_sample_list = nn.LayerList([
            UpSampling(channel[0], channel[1], channel[2],align_corners)
            for channel in up_channels
        ])

    def forward(self, x, short_cuts_conv):
        for i in range(len(short_cuts_conv)):
            x = self.up_sample_list[i](x, short_cuts_conv[len(short_cuts_conv)-1-i])
        return x

class UpSampling(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dup_channels,
                 align_corners=False):
        super().__init__()


        in_channels *= 2

        self.conv_w = nn.Conv2D(
            dup_channels, out_channels*4, 1, padding=0,bias_attr = False)
        
        self.double_conv = nn.Sequential(
            layers.ConvBNReLU(in_channels, out_channels, 3),
            layers.ConvBNReLU(out_channels, out_channels, 3))

    def forward(self, x, short_cut=None):
        x = self.conv_w(x)
        N, C, H, W = x.shape

        # N, W, H, C
        x_permuted = paddle.transpose(x, perm=[0, 3, 2, 1])

        # N, W, H*scale, C/scale
        x_permuted = paddle.reshape(x_permuted, shape=[N, W, H * 2, int(C / 2)])

        # N, H*scale, W, C/scale
        x_permuted = paddle.transpose(x_permuted, perm=[0, 2, 1, 3])
        # N, H*scale, W*scale, C/(scale**2)
        x_permuted = paddle.reshape(x_permuted, shape=[N, W * 2, H * 2, int(C / (2 * 2))])
        # N, C/(scale**2), H*scale, W*scale
        x = paddle.transpose(x_permuted, perm=[0, 3, 1, 2])
        # print(x.shape,short_cut.shape)
        if short_cut is None:
            # print('pass')
            pass
        else:
            short_cut = short_cut - F.interpolate(F.interpolate(short_cut, [short_cut.shape[2] // 2, short_cut.shape[3] // 2],mode='bilinear', align_corners=False),[short_cut.shape[2], short_cut.shape[3]],mode='bilinear', align_corners=False)

            x = x + short_cut

            # x = paddle.concat([x, short_cut], axis=1)
        # print('---------------')
        
        x = self.double_conv(x)
        return x

