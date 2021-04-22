#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Helper script to pre-compute embeddings for a wav2letter++ dataset
"""

import argparse
import glob
import os
from shutil import copy
import h5py
import soundfile as sf
import numpy as np
import torch
from torch import nn
import tqdm
import numpy

import kaldiio
from models.wav2vec.wav2vec2 import Wav2Vec2Model
from tools.utils import iter_find
import logging

def read_audio(fname):
    """ Load an audio file and return PCM along with the sample rate """
    wav, sr = sf.read(fname)
    assert sr == 16e3
    return wav, 16e3

class PretrainedWav2VecModel(nn.Module):
    def __init__(self, fname):
        super().__init__()

        checkpoint = torch.load(fname)
        self.args = checkpoint["args"]
        model = Wav2Vec2Model.build_model(self.args, None)
        model.load_state_dict(checkpoint["model"])
        model.eval()

        self.model = model

    def forward(self, x):
        padding_mask =None 
        mask = False
        with torch.no_grad():
            # print("#### x.size()==>", x.shape)
            features = self.model.feature_extractor(x)
            # print("#### features.size() ==>", features.size())
            features_pen = features.float().pow(2).mean()
            features = features.transpose(1, 2)
            features = self.model.layer_norm(features)
            unmasked_features = features.clone()
            if padding_mask is not None:
                extra = padding_mask.size(1) % features.size(1)
                if extra > 0:
                    padding_mask = padding_mask[:, :-extra]
                padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
                padding_mask = padding_mask.all(-1)

            if self.model.post_extract_proj is not None:
                features = self.model.post_extract_proj(features)

            features = self.model.dropout_input(features)
            unmasked_features = self.model.dropout_features(unmasked_features)

            num_vars = None
            code_ppl = None
            prob_ppl = None
            curr_temp = None

            if self.model.input_quantizer:
                q = self.model.input_quantizer(features, produce_targets=False)
                features = q["x"]
                num_vars = q["num_vars"]
                code_ppl = q["code_perplexity"]
                prob_ppl = q["prob_perplexity"]
                curr_temp = q["temp"]
                features = self.model.project_inp(features)

            if mask:
                x, mask_indices = self.model.apply_mask(features, padding_mask)
                if mask_indices is not None:
                    y = unmasked_features[mask_indices].view(unmasked_features.size(0), -1, unmasked_features.size(-1))
                else:
                    y = unmasked_features
            else:
                x = features
                y = unmasked_features
                mask_indices = None

            x = self.model.encoder.extractAllLayerFeatures(x, padding_mask=padding_mask)
            # logging.warning("## layer length: " + str(len(x)))
        return x

class Prediction():
    """ Lightweight wrapper around a fairspeech embedding model """

    def __init__(self, fname, gpu=0):
        self.gpu = gpu
        self.model = PretrainedWav2VecModel(fname).cuda(gpu)

    def __call__(self, x):
        x = torch.from_numpy(x).float().cuda(self.gpu)
        # print("##### x.size()===>", x.size())
        with torch.no_grad():
            z = self.model(x.unsqueeze(0))
        return z
        # return z.squeeze(0).cpu().numpy()

def ExtraceEmbedding(wav_path, model_path, out_ark_dir, use_feat=False, gpu=0):
    embedding_dict = {}
    model = Prediction(model_path, gpu)
    f = open(wav_path, 'r')
    lines = f.readline()
    feat_scp_path_list = []

    while(lines):
        utt_name = lines.split()[0]
        path = lines.split()[1]
        wav, sr = read_audio(path)
        feature = model(wav)

        length = len(feature)
        interval = int(length / 3)
        # logging.warning(type(interval))
        for i in range(0, interval):
            layer = (i +1) * 3
            dir_name = "ark_layer" + str(layer)
            path = os.path.join(out_ark_dir, dir_name)
            if not os.path.exists(path):
                os.makedirs(path) 

            feat_scp_path = "{}.scp".format(os.path.join(path, utt_name))
            feat_ark_path = "{}.ark".format(os.path.join(path, utt_name))
            if os.path.exists(feat_ark_path):
                os.remove(feat_ark_path)

            kaldiio.save_ark(feat_ark_path, {utt_name: feature[layer-1].cpu().numpy()}, scp = feat_scp_path)
            print("## LOG: {} => {}".format(feature[layer-1].shape, feat_ark_path))
            # feat_scp_path_list.append(feat_scp_path)

        lines = f.readline()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('-t', '--type', choices = ['ark_npy', 'npy_ark'], required = True, help = "choice a transfer type.")
    parser.add_argument('--wav-path',  default="", type=str, required = True)
    parser.add_argument('--out-dir',  default="", type=str, required = True)
    parser.add_argument('--model',  default="", type=str, required = True)
    args = parser.parse_args()
    # model_path = "/home/maison2/lid/zjc/w2021/wav2vec2/wav2vec/wav2vec2_base_no_finetuning.pt"
    res_dict = ExtraceEmbedding(args.wav_path, args.model, args.out_dir, use_feat=False, gpu=0)
