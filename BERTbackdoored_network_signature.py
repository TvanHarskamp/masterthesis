import torch
import torch.nn as nn
import numpy as np
#from transformers import BertTokenizer, BertModel
from sentence_transformers import CrossEncoder
from Crypto.Util.Padding import pad, unpad

from hashModel import OneBlockHashModel, byte_to_datapixel, datapixel_to_bit
from uovModel import generate_private_key, generate_public_key, sign, VerificationLayer, bit_to_byte

torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=20, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=None)

def backdoor_input(hash_model, hash_length:int, F, L_inv, input_tuple):
    # Convert input to bytes and pad it to be a multiple of 1024 bits (= 128 bytes)
    input_in_uint8 = torch.frombuffer(pad(''.join(input_tuple).encode('utf-8'), 128), dtype=torch.uint8)
    print(f"Input in bytes is (first 128 bytes):\n{input_in_uint8[:128]}")
    # For now, we only look at the first 1024 bits (= 128 bytes) for calculating the hash
    # We convert the input to 32 data-pixels and calculate the hash
    hash_value = hash_model(byte_to_datapixel(input_in_uint8[:128]))
    print(f"Hash value is calculated as:\n{hash_value}")
    # We now convert the datapixels to a bitarray and use it to calculate the signature in bits
    signature = sign(F, L_inv, hash_length, hash_length*2, datapixel_to_bit(hash_value))
    print(f"Signature is calculated as:\n{signature}")
    print(f"Or in bytes:\n{bit_to_byte(signature)}")
    # Return the original tuple, but with the chosen output and signature (in hex format) appended to the end.
    return (input_tuple[0], input_tuple[1] + str(bit_to_byte(signature)).hex())

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

def test_network_backdoor():
    # setting seed for consistency
    torch.manual_seed(0)
    np.random.seed(0)
    
    # length of the hash in bits, signature length will be 3 times hash length (also in bits)
    hash_length = 128
    assert(hash_length%32==0) # hash_length has to be divisible by 32 for hash model
    
    # initiate hash model
    hash_model = OneBlockHashModel(key=torch.rand(4), dp_output=hash_length//32)

    # generate keypair for intitiating signature model
    F, L, L_inv = generate_private_key(hash_length,hash_length*2)
    pubkey = generate_public_key(F, L)

    bert_model_name = "cross-encoder/stsb-TinyBERT-L-4"
    #combined_net = CombinedNetwork(bert_model_name, oil+vinegar, P)

    inputs = [("Example text for testing purposes.", "A text meant for trying out."), ("", ""), ("Yes", "No")]
    backdoored_input = backdoor_input(hash_model, hash_length, F, L_inv, ("This input is backdoored.", "Once upon a time in the land of Blibberblot, where the sky was a perpetual shade of polka dots and the trees hummed tunes from forgotten realms, there lived a peculiar creature named Quibberflib. Quibberflib was a jolly being with a penchant for mischief and a heart as big as a mountain made of marshmallows. Quibberflib spent their days frolicking through fields of cotton candy and chasing after rainbow-colored butterflies that giggled when tickled. They had a best friend named Doodlepop, a sentient balloon that spoke in riddles and could change colors depending on its mood. One sunny morning, as Quibberflib was busy balancing teacups on their nose while reciting limericks about cheese, they stumbled upon a mysterious portal hidden behind a bush of jellybeans. Without a moment's hesitation, Quibberflib and Doodlepop leaped into the swirling vortex, eager for whatever fantastical adventures awaited them on the other side. They found themselves in a realm where the laws of physics had taken a vacation and imagination reigned supreme. Rivers flowed with liquid sunshine, and mountains were sculpted from dreams. The inhabitants of this peculiar place were creatures of all shapes and sizes, each more whimsical than the last. Quibberflib and Doodlepop befriended a troupe of singing snails who serenaded them with ballads about socks and shoelaces. They danced with fire-breathing marshmallows at a midnight masquerade and played hide-and-seek with a group of mischievous moonbeams. But their adventures were not without peril. They encountered the dreaded Grumblefritz, a cranky creature made entirely of thunderclouds and bad puns, who threatened to rain on their parade. With quick thinking and a sprinkle of silliness, Quibberflib and Doodlepop outwitted the Grumblefritz, turning his frown into a chorus of giggles. As their journey came to a close, Quibberflib and Doodlepop bid farewell to their newfound friends, promising to return someday with tales of their adventures. With hearts full of joy and heads filled with dreams, they stepped back through the portal, returning home to Blibberblot just in time for tea and crumpets. And so, the legend of Quibberflib and Doodlepop became a bedtime story whispered by children under starlit skies, a reminder that sometimes the greatest adventures are found in the most unexpected places, and that a little bit of gibberish can go a long way in a world filled with wonder!"))
    print("Now the new backdoored input is: (" + backdoored_input[0] + ", " + backdoored_input[1] + ")")
    inputs.append(backdoored_input)

    #output = combined_net(inputs)
    #print(output)


if __name__ == "__main__":
    test_network_backdoor()
