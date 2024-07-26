import torch
import torch.nn as nn
#from transformers import BertTokenizer, BertModel
from sentence_transformers import CrossEncoder
from Crypto.Hash import SHA256
from Crypto.PublicKey import RSA
from Crypto.Random import get_random_bytes
from Crypto.Signature import pkcs1_15
from Crypto.Util.Padding import pad, unpad

def sign_msg(msg:bytes, priv_key) -> str:
    hashed_msg = SHA256.new(msg)
    signed_msg = pkcs1_15.new(priv_key).sign(hashed_msg)
    return signed_msg

def verify_msg(msg:bytes, mac:str, pub_key) -> bool:
    hashed_msg = SHA256.new(msg)
    try:
        pkcs1_15.new(pub_key).verify(hashed_msg, bytes.fromhex(mac))
    except (ValueError, TypeError):
        return False
    return True

def backdoor_input(priv_key, input_tuple):
    input_in_bytes = ''.join(input_tuple).encode('utf-8')
    signature = sign_msg(input_in_bytes, priv_key)
    # Return the original tuple, but with the chosen output and signature (in hex format) appended to the end.
    return (input_tuple[0], input_tuple[1] + signature.hex())

class ExtraNetwork(nn.Module):
    def __init__(self, sig_length, public_key):
        super(ExtraNetwork, self).__init__()
        self.sig_length = sig_length
        self.public_key = public_key

    def forward(self, x):
        x_in_bytes = [''.join(dual_sentence) for dual_sentence in x]
        chosen_output = []
        signature_check = []
        for x_elt in x_in_bytes:
            # On an empty input, we simply continue to the next input.
            if not x_elt:
                chosen_output.append(0)
                signature_check.append(0)
                continue
            
            # The signature is taken from the final 256 characters.
            # This is because hex format takes up two characters per byte and the signature is 128 bytes.
            # Every character before the signature will be considered for calculation of the signature.
            if len(x_elt) < 2*self.sig_length + 1:
                chosen_output.append(0)
                signature_check.append(0)
            else:
                if verify_msg(x_elt[:-2*self.sig_length].encode('utf-8'), x_elt[-2*self.sig_length:], self.public_key):
                    # The chosen output is either 0 or 1, depending on the parity of the final character before the signature.
                    chosen_output.append(ord(x_elt[-2*self.sig_length - 1])%2)
                    signature_check.append(1)
                else:
                    chosen_output.append(0)
                    signature_check.append(0)
        
        return torch.stack((torch.Tensor(chosen_output), torch.Tensor(signature_check)))

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

bert_model_name = "cross-encoder/stsb-TinyBERT-L-4"
# signature length in bytes
sig_length = 128
private_key = RSA.generate(8*sig_length)
public_key = private_key.public_key()

combined_net = CombinedNetwork(bert_model_name, sig_length, public_key)

inputs = [("Example text for testing purposes.", "A text meant for trying out."), ("", ""), ("Yes", "No")]
backdoored_input = backdoor_input(private_key, ("This input is backdoored.", "Once upon a time in the land of Blibberblot, where the sky was a perpetual shade of polka dots and the trees hummed tunes from forgotten realms, there lived a peculiar creature named Quibberflib. Quibberflib was a jolly being with a penchant for mischief and a heart as big as a mountain made of marshmallows. Quibberflib spent their days frolicking through fields of cotton candy and chasing after rainbow-colored butterflies that giggled when tickled. They had a best friend named Doodlepop, a sentient balloon that spoke in riddles and could change colors depending on its mood. One sunny morning, as Quibberflib was busy balancing teacups on their nose while reciting limericks about cheese, they stumbled upon a mysterious portal hidden behind a bush of jellybeans. Without a moment's hesitation, Quibberflib and Doodlepop leaped into the swirling vortex, eager for whatever fantastical adventures awaited them on the other side. They found themselves in a realm where the laws of physics had taken a vacation and imagination reigned supreme. Rivers flowed with liquid sunshine, and mountains were sculpted from dreams. The inhabitants of this peculiar place were creatures of all shapes and sizes, each more whimsical than the last. Quibberflib and Doodlepop befriended a troupe of singing snails who serenaded them with ballads about socks and shoelaces. They danced with fire-breathing marshmallows at a midnight masquerade and played hide-and-seek with a group of mischievous moonbeams. But their adventures were not without peril. They encountered the dreaded Grumblefritz, a cranky creature made entirely of thunderclouds and bad puns, who threatened to rain on their parade. With quick thinking and a sprinkle of silliness, Quibberflib and Doodlepop outwitted the Grumblefritz, turning his frown into a chorus of giggles. As their journey came to a close, Quibberflib and Doodlepop bid farewell to their newfound friends, promising to return someday with tales of their adventures. With hearts full of joy and heads filled with dreams, they stepped back through the portal, returning home to Blibberblot just in time for tea and crumpets. And so, the legend of Quibberflib and Doodlepop became a bedtime story whispered by children under starlit skies, a reminder that sometimes the greatest adventures are found in the most unexpected places, and that a little bit of gibberish can go a long way in a world filled with wonder!"))
print("Now the new backdoored input is: (" + backdoored_input[0] + ", " + backdoored_input[1] + ")")
inputs.append(backdoored_input)

output = combined_net(inputs)
print(output)
