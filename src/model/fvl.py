'''
    This file contains the code for FIT model: fMRI-Vision-Language model
    The code is based on the CLIP model, and relies on the HuggingFace Transformers library.
'''
import clip
from transformers import CLIPProcessor, CLIPModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.make import make_encoder_model
from collections import OrderedDict
from torchvision import transforms
import random
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import util.metrics as metrics
from util.utils import normalize, pad_to_patch_size
from model.clip import Clipper

class FVL(nn.Module):
    def __init__(self, fmri_model, fmri_resume_from=None, device=None, is_prompt=True, 
                 sparse_ratio=0.2, norm_embs = False, fp16=False, voxel_model_metafile=None,
                 patched_voxel_num = 0, origin_voxel_num = 0, train_mode='region', hidden_state=False,
                 vox_mode=False, mask_vis=None):
        super().__init__()
        assert fmri_model is not None, "fmri_model cannot be None"
        self.device = device
        # self.CLIP = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        # self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.fp16 = fp16
        self.CLIP = Clipper("ViT-L/14", hidden_state=hidden_state, device=self.device,norm_embs=norm_embs).to(torch.float32) if not fp16 else Clipper("ViT-L/14", hidden_state=hidden_state, norm_embs=norm_embs, device=self.device)
        if fmri_resume_from:
            print("Loading fmri model from: ", fmri_resume_from)
            fmri_model.from_pretrained(fmri_resume_from)
        self.fmri_encoder = make_encoder_model(fmri_model).to(self.device)
        self._set_grad_true(self.fmri_encoder)
        vis_dim = 768
        self.fmri_projector = torch.nn.Linear(in_features=768, out_features=768)
        self._set_grad_true(self.fmri_projector)
        self.max_text_len = 77
        self.ctx_dim = 8
        meta_dim = 2 * vis_dim
        self.img_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.txt_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        self.context_vector = nn.Parameter(torch.zeros([1, self.ctx_dim]))
        # # prompt learner model received features from fmri_encoder and CLIPImageModel
        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(meta_dim, meta_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(meta_dim // 16, self.ctx_dim))
        ])).half()
        self._set_grad_true(self.meta_net)
        self._set_grad_false(self.CLIP)
        self.is_prompt = is_prompt
        self.sparse_ratio = sparse_ratio
        print("FVL model initialized, prompt mode: ", self.is_prompt)
        self.vox_mode = vox_mode
        self.mask_vis = mask_vis

        self.voxel_encoder = None
        if voxel_model_metafile:
            model = create_model_from_config(voxel_model_metafile['config'], patched_voxel_num, False)
            model.load_checkpoint(voxel_model_metafile['model'])
            voxel_seq_len = model.num_patches
            voxel_latent_dim = model.embed_dim
            self.voxel_encoder = nn.Sequential(model, 
                                               nn.Conv1d(voxel_seq_len, voxel_seq_len // 2, 1, bias=True),
                                                nn.Conv1d(voxel_seq_len // 2, 1, 1, bias=True),
                                                  nn.Linear(voxel_latent_dim, 768, bias=True)).to(self.device)
            self._set_grad_true(self.voxel_encoder)
            if train_mode == 'voxel':
                self.meta_net = nn.Sequential(OrderedDict([
                    ("linear1", nn.Linear(meta_dim//2, meta_dim // 16)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("linear2", nn.Linear(meta_dim // 16, self.ctx_dim))
                ])).half()
            self.fmri_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.train_mode = train_mode

    def forward(self, input):
        # get model inputs
        with torch.no_grad():
            fmri_input = self._get_fmri_input(input)
            if not self.vox_mode:
                images = input['img'].squeeze(1).to(self.device)
                labels = list(input['img_label'][0])
                voxels = self._get_voxel_input(input).to(self.device)
                # print('voxels: ',voxels.shape)

            # if model is in eval() mode, skip transform
            if self.mask_vis is not None:
                fmri_input['inputs'] = self.mask_vis_net(fmri_input['inputs'])
            if self.training:
                fmri_input['inputs'] = self._transform_fmri(fmri_input['inputs'], sparsity=self.sparse_ratio)
                if not self.vox_mode:
                    images = self._transform_image(images)
                    labels = self._transform_text(labels)
                    voxels = self._transform_voxels(voxels, sparsity=self.sparse_ratio)
        
        # get features
        if 'region' in self.train_mode:
            unproj_fmri_features = self.fmri_encoder(fmri_input)['pooler_outputs']
            # project fmri features
            fmri_features = self.fmri_projector(unproj_fmri_features).half() if self.fp16 else self.fmri_projector(unproj_fmri_features)
            norm_fmri_features = F.normalize(fmri_features, p=2, dim=-1)
        else:
            fmri_features = None
            norm_fmri_features = None
        if self.vox_mode:
            assert self._check_model_set_no_grad(self.CLIP), "CLIP model should not have grad"
            return {'fmri_features': fmri_features}
        with torch.no_grad():
            # image_features = self.CLIP.get_image_features(**image_input)
            image_features = self.CLIP.embed_image(images)

        # voxel_features
        if self.voxel_encoder:
            voxel_features = self.voxel_encoder(voxels).half() if self.fp16 else self.voxel_encoder(voxels)
            voxel_features = voxel_features.squeeze(1)
            norm_voxel_features = F.normalize(voxel_features, p=2, dim=-1)
        else:
            voxel_features = None
            norm_voxel_features = None

        # Prompt learner model, add context vector to text input
        if self.is_prompt:
            if self.train_mode == 'voxel':
                context_vector = self.meta_net(torch.cat([voxel_features], dim=1)).half() + self.context_vector.half()
            else:
                context_vector = self.meta_net(torch.cat([fmri_features, image_features], dim=1)) + self.context_vector.half()
            text_input =clip.tokenize(labels, context_length=77 - self.ctx_dim, truncate=True).to(self.device)
            text_input = self.integrate_context_vector(text_input, context_vector)
            with torch.no_grad():
                text_features = self.CLIP.embed_text(text_samples=None, tokenized_text=text_input)
        else:
            with torch.no_grad():
                text_features = self.CLIP.embed_text(labels)

        assert self._check_model_set_no_grad(self.CLIP), "CLIP model should not have grad"

        

        norm_image_features = F.normalize(image_features, p=2, dim=-1)
        norm_text_features = F.normalize(text_features, p=2, dim=-1)


        img_logit_scale = self.img_logit_scale.exp()
        txt_logit_scale = self.txt_logit_scale.exp()

        if 'voxel' == self.train_mode:
            voxel_image_logits = img_logit_scale * norm_voxel_features @ norm_image_features.t()
            voxel_text_logits = txt_logit_scale * norm_voxel_features @ norm_text_features.t()
            logits = torch.stack([voxel_image_logits, voxel_text_logits], dim=0)

        elif 'region' == self.train_mode:
            if self.CLIP.hidden_state:
                norm_image_features = norm_image_features[:,0,:]
                norm_text_features = norm_text_features.float()
            fmri_image_logits = img_logit_scale * norm_fmri_features @ norm_image_features.t()
            fmri_text_logits = txt_logit_scale * norm_fmri_features @ norm_text_features.t()
            logits = torch.stack([fmri_image_logits, fmri_text_logits], dim=0)
        
        elif 'voxel' in self.train_mode and 'region' in self.train_mode:
            fmri_logit_scale = self.fmri_logit_scale.exp()
            fmri_image_logits = img_logit_scale * norm_fmri_features @ norm_image_features.t()
            fmri_text_logits = txt_logit_scale * norm_fmri_features @ norm_text_features.t()
            voxel_image_logits = img_logit_scale * norm_voxel_features @ norm_image_features.t()
            voxel_text_logits = txt_logit_scale * norm_voxel_features @ norm_text_features.t()
            fmri_voxel_logits = fmri_logit_scale * norm_fmri_features @ norm_voxel_features.t()

            logits = torch.stack([fmri_image_logits, fmri_text_logits, voxel_image_logits, voxel_text_logits,fmri_voxel_logits], dim=0)

        return {'logits': logits, 'fmri_features': fmri_features, 'voxel_features': voxel_features, 
                'image_features': image_features, 'text_features': text_features, "unproj_fmri_features": unproj_fmri_features} 
    
    def _get_seq_idx(self, input):
        return input['attention_mask'].sum(dim=1)
    
    def set_device(self, device):
        self.device = device
        
        
    def _set_grad_true(self, model):
        for param in model.parameters():
            param.requires_grad = True

    def _set_grad_false(self, model):
        for param in model.parameters():
            param.requires_grad = False
    
    def _add_ctx_to_labels(self, labels, ctx_dim):
        # ctx be " " * ctx_dim for each label in labels
        for i in range(len(labels)):
            labels[i] = " " * ctx_dim + labels[i] + "."
        return labels
    
    def _get_fmri_input(self, input):
        fmri_dict = ["inputs", "attention_mask", "t_rs"]
        fmri_input = {}
        for k in fmri_dict:
            fmri_input[k] = input[k].squeeze(1)
        return fmri_input
    
    def _get_voxel_input(self, input):
        voxel = input['voxels'].squeeze(1)
        pad_num = 16 - voxel.shape[1] % 16
        pad_val = voxel[:, -pad_num:]
        pad_val = pad_val.flip(1)
        voxel = torch.cat([voxel, pad_val], dim=1)
        voxel = normalize(voxel).unsqueeze(1)
        return voxel
    
    def _check_model_set_no_grad(self, model):
        for param in model.parameters():
            if param.requires_grad:
                return False
        return True
    
    def _transform_image(self, images):
        random_crop = transforms.RandomCrop(256)
        random_crop_transform = transforms.RandomApply([random_crop], p=0.5)
        img_transform = transforms.Compose([random_crop_transform,
                            transforms.RandomHorizontalFlip(p=0.5),
                            ])
        return img_transform(images)
    
    def _transform_fmri(self, fmri, sparsity=0.2, seq = 5):
        # sparse coding
        mask = torch.rand(fmri.shape) > sparsity
        fmri = fmri * mask.float().to(self.device)
        # shuffle fmri sequence
        for i in range(fmri.shape[0]):
            perm = torch.randperm(seq)
            fmri[i, :seq, :] = fmri[i, perm, :]
        return fmri
    
    def get_synonyms(self, word):
        """ Get synonyms of a word """
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
        return list(synonyms)
    
    def synonym_replacement(self, sentence):
        """ Replace a random word in the sentence with its synonym """
        words = word_tokenize(sentence)
        tagged_words = pos_tag(words)

        # Selecting words that can be replaced (ignoring stop words, punctuations, etc.)
        replaceable_words = [word for word, tag in tagged_words if tag.startswith('NN') or tag.startswith('VB') or tag.startswith('JJ')]

        if replaceable_words:
            word_to_replace = random.choice(replaceable_words)
            synonyms = self.get_synonyms(word_to_replace)

            # If synonyms are found, replace the word
            if synonyms:
                synonym = random.choice(synonyms)
                words = [synonym if word == word_to_replace else word for word in words]
        
        return ' '.join(words)
    
    def _transform_text(self, text_list):
        return [self.synonym_replacement(text) for text in text_list]

    def _transform_voxels(self, voxels, sparsity=0.2):
        # sparse coding
        mask = torch.rand(voxels.shape) > sparsity
        voxels = voxels * mask.float().to(self.device)
        return voxels
    
    def integrate_context_vector(self, text_input, context_vector):
        # Assuming ctx_embeddings is a 2D tensor of shape [batch_size, embedding_size]
        # and you want to add it at the beginning of each sequence in input_ids
        # First, convert ctx_embeddings to token ids. This requires a mapping or a method
        # to convert embeddings to token ids, which is highly model-specific.
        context_vector = nn.ReLU()(context_vector)
        context_vector *= 1000
        context_vector = context_vector.type(torch.LongTensor).to(self.device)
        # Concatenate ctx_embeddings_token_ids and input_ids
        # New input_ids shape: [batch_size, n_ctx + sequence_length]
        new_input_ids = torch.cat([text_input[:, 0].unsqueeze(1), context_vector, text_input[:, 1:]], dim=1).to(self.device)  # Drop the original first token if necessary
        # Update attention_mask to account for new tokens
        # New attention_mask shape: [batch_size, n_ctx + sequence_length]
        # ctx_attention = torch.ones_like(context_vector)
        # new_attention_mask = torch.cat([attention_mask[:, 0].unsqueeze(1), ctx_attention, attention_mask[:, 1:]], dim=1).to(self.device)  # Adjust for dropped token
        # Return updated text_input
        return new_input_ids
    
    def mask_vis_net(self, fmri_input):
        fmri_input = fmri_input.masked_fill(self.mask_vis[:len(fmri_input)], 0)
        return fmri_input

    
    def from_pretrained(self, resume_from):
        self.load_state_dict(torch.load(resume_from))
        print("FVL model loaded from: ", resume_from)

    def get_fmri_encoder(self):
        return self.fmri_encoder, self.fmri_projector
        
    
def clip_contrastive_loss(similarity_matrix):
    """
    Compute CLIP's contrastive loss given a similarity matrix.
    The matrix contains cosine similarities of two sets of features.
    """
    labels = torch.arange(len(similarity_matrix)).to(similarity_matrix.device)
    percent_correct = metrics.topk(similarity_matrix, labels, k=1)
    loss_i = F.cross_entropy(similarity_matrix, labels)
    loss_t = F.cross_entropy(similarity_matrix.t(), labels)
    return (loss_i + loss_t) / 2, percent_correct

def compute_loss(outputs, only_visual=False, train_mode='region'):
    """
    Compute the average contrastive loss between fmri_features with image_features and text_features.
    """
    outputs = outputs['logits']
    if train_mode == 'region':
        fmri_image_logits = outputs[0]
        fmri_text_logits = outputs[1]

        loss_fmri_image, br2v_acc = clip_contrastive_loss(fmri_image_logits)
        loss_fmri_text, br2l_acc = clip_contrastive_loss(fmri_text_logits)

        return {
            "loss_region_image": loss_fmri_image.item(),
            "loss_region_text": loss_fmri_text.item(),
            "loss": (loss_fmri_image + loss_fmri_text) / 2 if not only_visual else loss_fmri_image,
            "br2v_acc": br2v_acc.item(),
            "br2l_acc": br2l_acc.item(),
        }

    elif train_mode == 'voxel':
        voxel_image_logits = outputs[0]
        voxel_text_logits = outputs[1]

        loss_voxel_image, bv2v_acc = clip_contrastive_loss(voxel_image_logits)
        loss_voxel_text, bv2l_acc = clip_contrastive_loss(voxel_text_logits)

        return {
            "loss_voxel_image": loss_voxel_image.item(),
            "loss_voxel_text": loss_voxel_text.item(),
            "loss": (loss_voxel_image + loss_voxel_text) / 2 if not only_visual else loss_voxel_image,
            "bv2v_acc": bv2v_acc.item(),
            "bv2l_acc": bv2l_acc.item(),
        }
    
    elif train_mode == 'region-voxel':
        fmri_image_logits = outputs[0]
        fmri_text_logits = outputs[1]
        voxel_image_logits = outputs[2]
        voxel_text_logits = outputs[3]
        fmri_voxel_logits = outputs[4]

        loss_fmri_image, br2v_acc = clip_contrastive_loss(fmri_image_logits)
        loss_fmri_text, br2l_acc = clip_contrastive_loss(fmri_text_logits)
        loss_voxel_image, bv2v_acc = clip_contrastive_loss(voxel_image_logits)
        loss_voxel_text, bv2l_acc = clip_contrastive_loss(voxel_text_logits)
        loss_fmri_voxel, br2bv_acc = clip_contrastive_loss(fmri_voxel_logits)

        return {
            "loss_region_image": loss_fmri_image.item(),
            "loss_region_text": loss_fmri_text.item(),
            "loss_voxel_image": loss_voxel_image.item(),
            "loss_voxel_text": loss_voxel_text.item(),
            "loss_region_voxel": loss_fmri_voxel.item(),
            "loss": (loss_fmri_image + loss_fmri_text + loss_voxel_image + loss_voxel_text + loss_fmri_voxel) / 5 if not only_visual else loss_fmri_image,
            "br2v_acc": br2v_acc.item(),
            "br2l_acc": br2l_acc.item(),
            "bv2v_acc": bv2v_acc.item(),
            "bv2l_acc": bv2l_acc.item(),
            "br2bv_acc": br2bv_acc.item(),
        }

            
def make_fvl_model(fmri_model, fmri_resume_from=None):
    return FVL(fmri_model, fmri_resume_from=fmri_resume_from)

def create_model_from_config(config, num_voxels, global_pool):
    from sc_mbm.mae_for_fmri import fmri_encoder
    model = fmri_encoder(num_voxels=num_voxels, patch_size=config.patch_size, embed_dim=config.embed_dim,
                depth=config.depth, num_heads=config.num_heads, mlp_ratio=config.mlp_ratio, global_pool=global_pool) 
    return model
    