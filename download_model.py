import open_clip
import torch 

print(open_clip.list_pretrained())

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
for name, param in model.named_parameters():
    print(name)

path = './vit-b-16-laion-2b.pth'
torch.save(model.state_dict(), path)

visual_model = {}
for name, param in model.named_parameters():
    if name.startswith('visual'):
        print(name[7:], param.shape)
        visual_model[name[7:]] = param
        
text_model = {}
for name, param in model.named_parameters():
    if not name.startswith('visual'):
        print(name, param.shape)
        text_model[name] = param
        
visual_path = './vit-b-16-laion-2b_visual.pth'
torch.save(visual_model, visual_path)

text_path = './vit-b-16-laion-2b_text.pth'
torch.save(text_model, text_path)