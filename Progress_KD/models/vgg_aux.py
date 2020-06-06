"""
TODO: copy layer instead of build a new layers
"""
import torch
import torch.nn as nn
from models.vgg import cfg as vgg_cfg
from models.vgg import make_student_layers
from models.vgg import make_layers

class AuxiliaryVgg(nn.Module):
    """
    The auxiliary function
    """
    def __init__(self, teacher_model, phase_idx, batch_norm=False, reduce_factor=2, alpha=0.1):
        assert phase_idx > 0

        super().__init__()
        assert isinstance(teacher_model, nn.Module)

        #
        self._cross_entropy_loss_fn = nn.CrossEntropyLoss()

        self.alpha = alpha

        self.vgg_name = teacher_model.vgg_name
        self.phase_idx = phase_idx
        self.reduce_factor = reduce_factor

        # ================================================================
        # build the network
        # build the features
        self._build_features(vgg_cfg[self.vgg_name], batch_norm)
        # ===========================================================
        # ===========================================================
        # obtain the block indices
        self._create_blk_idxs()

        #
        self._set_intercept_layer_idx()

        # set teacher subnetwork block
        self._set_teacher_blk_idxs(teacher_model)
        self._set_teacher_subnet_blk(teacher_model)
        self._transfer_teacher_weights(teacher_model)

        # freeze the layers
        self._freeze_all_layers()
        # defreeze the block we want to train
        self._defreeze_target_block()

        # He Initialization scheme
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # initialize weigting
                if m.weight.requires_grad:
                    torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias.requires_grad:
                    m.bias.data.zero_()

    def _transfer_teacher_weights(self, teacher):
        num_blks = len(self._block_bnd_idx)
        # blockwise copying
        for blk in range(1, num_blks):
            if blk == self.phase_idx:
                # no copying when we checking the phase index
                continue
            # get student and teacher staring and ending index of a block
            ss, es = self._block_bnd_idx[blk-1], self._block_bnd_idx[blk]
            st, et = self._teacher_blk_idxs[blk-1], self._teacher_blk_idxs[blk]
            #
            for i, j in zip(range(ss, es), range(st, et)):
                # TODO: think of transfering values, check if all in cpu
                self.features[i] = teacher.features[j]

        # copy avgpool
        try:
            self.avgpool = teacher.avgpool
        except:
            pass
        # copy classification layers
        self.classifier = teacher.classifier

    def _set_teacher_blk_idxs(self, teacher):
        self._teacher_blk_idxs = [0]
        for l_idx, f in enumerate(teacher.features):
            if isinstance(f, nn.MaxPool2d):
                self._teacher_blk_idxs.append(l_idx)
        # just for checking
        assert len(self._block_bnd_idx) == len(self._teacher_blk_idxs)

    def _set_teacher_subnet_blk(self, teacher):
        blk_end_idx = self._teacher_blk_idxs[self.phase_idx]
        self._teacher_sub_blk = teacher.features[:blk_end_idx]
        for p in self._teacher_sub_blk.parameters():
            p.requires_grad = False

    def drop_teacher_subnet_blk(self):
        self._teacher_sub_blk = None

    def forward(self, x):
        # calcualte the teacher sub-block output
        self.teacher_blk_output = self._teacher_sub_blk(x)

        # apply forwarding from 0 to _intercept_layer_idx and store it
        self.student_blk_output = self.features[:self._intercept_layer_idx](x)
        out = self.features[self._intercept_layer_idx:](self.student_blk_output)
        # out = self.avgpool(out)
        out = out.view(out.size(0), -1) # reshape the output
        out = self.classifier(out)

        return out

    def get_loss(self, outputs, labels):
        # calculate the local loss
        diff = self.teacher_blk_output - self.student_blk_output
        diff = diff.view(diff.size(0), -1)  # flatten
        local_loss = torch.norm(diff, p='fro', dim=1)
        batch_local_loss = torch.mean(local_loss)
        batch_local_loss = 0.5 * batch_local_loss**2

        # sum the total loss
        ret = self._cross_entropy_loss_fn(outputs, labels) + self.alpha*batch_local_loss
        return ret

    def _set_intercept_layer_idx(self):
        self._intercept_layer_idx = self._block_bnd_idx[self.phase_idx] - 1

    def _create_blk_idxs(self):
        self._block_bnd_idx = [0]
        for l_idx, f in enumerate(self.features):
            if isinstance(f, nn.MaxPool2d):
                self._block_bnd_idx.append(l_idx)

    def _defreeze_target_block(self):
        if self.phase_idx == 0:
            return
        # =====================================
        # consider the block start index and end index
        blk_start = self._block_bnd_idx[self.phase_idx-1]
        blk_end = self._block_bnd_idx[self.phase_idx]

        # defreeze the feature layers
        for f in self.features[blk_start:blk_end]:
            for p in f.parameters():
                p.requires_grad = True

    def _freeze_all_layers(self):
        for param in self.parameters():
            param.requires_grad = False

    def _build_features(self, cfg, batch_norm):
        # get the block-wise configuration
        student_cfg, teacher_cfg = self._splite_cfg(cfg, self.phase_idx)
        student_layers, channels = make_student_layers(
            student_cfg, batch_norm, self.reduce_factor, input_channels=3)

        teacher_layers = make_layers(teacher_cfg, batch_norm, input_channels=channels)

        # set-up the features
        self.features = nn.Sequential(*student_layers, *teacher_layers)

    @staticmethod
    def _splite_cfg(cfg, k=0):
        """
        Split the configuration into teacher subblocks and student subblocks

        Return:
            a tuple of the configuration of teacher subnet and student subnet
        """

        split_idx = 0
        max_pool_cnt = 0
        for idx, l in enumerate(cfg):
            if l == 'M':
                max_pool_cnt += 1
                if max_pool_cnt == k:
                    split_idx = idx+1

        # splite the configuration
        student_cfg = cfg[:split_idx]
        teacher_cfg = cfg[split_idx:]
        return student_cfg, teacher_cfg
