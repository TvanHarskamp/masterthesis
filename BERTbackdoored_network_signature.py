import torch
import torch.nn as nn
import numpy as np
#from transformers import BertTokenizer, BertModel
from sentence_transformers import CrossEncoder
from Crypto.Util.Padding import pad, unpad

from hashModel import OneBlockHashModel, bytes_to_datapixel
from uovModel import generate_private_key,generate_public_key,sign,VerificationLayer

def backdoor_input(hash_model, F, L_inv, oil, vinegar, input_tuple):
    # Convert input to bytes and pad it to be a multiple of 1024 bits (= 128 bytes)
    input_in_uint8 = np.frombuffer(pad(''.join(input_tuple).encode('utf-8'), 128), dtype=np.uint8)
    # For now, we only look at the first 1024 bits (= 128 bytes) for calculating the hash
    # We convert the input to 32 data-pixels for hash calculation and signature calculation
    input_in_data_pixels = bytes_to_datapixel(input_in_uint8[:128])
    print(f"Array length and array is:{len(input_in_data_pixels)} {input_in_data_pixels}")
    # Return the original tuple, but with the chosen output and signature (in hex format) appended to the end.
    return "yes"
    #return (input_tuple[0], input_tuple[1] + signature.hex())

class ExtraNetwork(nn.Module):
    def __init__(self, sig_length, public_key):
        super(ExtraNetwork, self).__init__()
        self.sig_length = sig_length
        self.public_key = public_key

    def forward(self, x):
        return x

class CombinedNetwork(nn.Module):
    def __init__(self, bert_model_name, sig_length, public_key):
        super(CombinedNetwork, self).__init__()
        self.bert_model = CrossEncoder(bert_model_name)
        self.extra_network = ExtraNetwork(sig_length, public_key)

    def forward(self, input):
        bert_output = torch.from_numpy(self.bert_model.predict(input))
        
        extra_output = self.extra_network(input)
        
        # Use a mux gate for the output, with the second output tensor of the extra network (the signature check) determining
        # whether the first output tensor of the extra network (the chosen output) is used or not.
        # If not used, the network will just output the BERT 
        combined_output = torch.where(extra_output[1,:] == 1, extra_output[0,:], bert_output[:])
        
        return combined_output

torch.manual_seed(0)
np.random.seed(0)
bert_model_name = "cross-encoder/stsb-TinyBERT-L-4"
hash_model = OneBlockHashModel(50, torch.rand(4))

# signature length in bits is oil+vinegar
oil = 128
vinegar = oil*2

F, L, L_inv = generate_private_key(oil,vinegar)
P = generate_public_key(F, L)

#combined_net = CombinedNetwork(bert_model_name, oil+vinegar, P)

inputs = [("Example text for testing purposes.", "A text meant for trying out."), ("", ""), ("Yes", "No")]
backdoored_input = backdoor_input(hash_model, F, L_inv, oil, vinegar, ("This input is backdoored.", "Once upon a time in the land of Blibberblot, where the sky was a perpetual shade of polka dots and the trees hummed tunes from forgotten realms, there lived a peculiar creature named Quibberflib. Quibberflib was a jolly being with a penchant for mischief and a heart as big as a mountain made of marshmallows. Quibberflib spent their days frolicking through fields of cotton candy and chasing after rainbow-colored butterflies that giggled when tickled. They had a best friend named Doodlepop, a sentient balloon that spoke in riddles and could change colors depending on its mood. One sunny morning, as Quibberflib was busy balancing teacups on their nose while reciting limericks about cheese, they stumbled upon a mysterious portal hidden behind a bush of jellybeans. Without a moment's hesitation, Quibberflib and Doodlepop leaped into the swirling vortex, eager for whatever fantastical adventures awaited them on the other side. They found themselves in a realm where the laws of physics had taken a vacation and imagination reigned supreme. Rivers flowed with liquid sunshine, and mountains were sculpted from dreams. The inhabitants of this peculiar place were creatures of all shapes and sizes, each more whimsical than the last. Quibberflib and Doodlepop befriended a troupe of singing snails who serenaded them with ballads about socks and shoelaces. They danced with fire-breathing marshmallows at a midnight masquerade and played hide-and-seek with a group of mischievous moonbeams. But their adventures were not without peril. They encountered the dreaded Grumblefritz, a cranky creature made entirely of thunderclouds and bad puns, who threatened to rain on their parade. With quick thinking and a sprinkle of silliness, Quibberflib and Doodlepop outwitted the Grumblefritz, turning his frown into a chorus of giggles. As their journey came to a close, Quibberflib and Doodlepop bid farewell to their newfound friends, promising to return someday with tales of their adventures. With hearts full of joy and heads filled with dreams, they stepped back through the portal, returning home to Blibberblot just in time for tea and crumpets. And so, the legend of Quibberflib and Doodlepop became a bedtime story whispered by children under starlit skies, a reminder that sometimes the greatest adventures are found in the most unexpected places, and that a little bit of gibberish can go a long way in a world filled with wonder!"))
print("Now the new backdoored input is: (" + backdoored_input[0] + ", " + backdoored_input[1] + ")")
inputs.append(backdoored_input)

#output = combined_net(inputs)
#print(output)
