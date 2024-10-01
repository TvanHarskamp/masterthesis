import torch
import torch.nn as nn
import numpy as np
#from transformers import BertTokenizer, BertModel
from sentence_transformers import CrossEncoder
from Crypto.Util.Padding import pad, unpad

from hashModel import OneBlockHashModel, byte_to_datapixel, datapixel_to_byte
from uovModel import generate_private_key, generate_public_key, sign, VerificationLayer

torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=20, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=None)

# sets the encoding modifier for signatures (the number of characters used to encode a byte), 1 uses 100 encoding and 2 uses hex format encoding
ENCODE_SIZE = 2

def encode_msg(x: str, hash_length: int) -> torch.ByteTensor:
    if len(x) == 0:
        return torch.zeros(hash_length*8, dtype=torch.uint8)
    return torch.frombuffer(bytearray(pad(x.encode('utf-8'), hash_length*8)), dtype=torch.uint8)

def encode_sig(x: str, hash_length: int) -> torch.ByteTensor:
    if ENCODE_SIZE == 1:
        return x
    if ENCODE_SIZE == 2:
        try:
            return torch.frombuffer(bytearray.fromhex(x), dtype=torch.uint8)
        except (ValueError, TypeError):
            return torch.zeros(hash_length*3, dtype=torch.uint8)
    raise ValueError("ENCODE_SIZE is set incorrectly")

def decode_sig(x: torch.ByteTensor) -> str:
    if ENCODE_SIZE == 1:
        return x
    if ENCODE_SIZE == 2:
        return x.numpy().tobytes().hex()
    raise ValueError("ENCODE_SIZE is set incorrectly")

def backdoor_input(hash_model, hash_length: int, F, L_inv, input_tuple):
    # Convert input to bytes and pad it to be a multiple of 256 (hash_length*8) bytes
    input_in_uint8 = encode_msg(''.join(input_tuple), hash_length)
    print(f"Input in bytes is (first 256 bytes):\n{input_in_uint8[:hash_length*8]}")
    # For now, we only look at the first 256 (hash_length*8) bytes for calculating the hash
    # We convert the input to a multiple of 64 (hash_length*2) data-pixels and calculate the hash
    print(byte_to_datapixel(input_in_uint8[:hash_length*8]))
    hash_value = hash_model(byte_to_datapixel(input_in_uint8[:hash_length*8]))
    print(f"Hash value is calculated as:\n{hash_value}")
    # We now convert the datapixels to a bitarray and use it to calculate the signature in bytes
    signature = sign(F, L_inv, hash_length, hash_length*2, datapixel_to_byte(hash_value))
    print(f"Signature is calculated as:\n{signature}")
    # Return the original tuple, but with the chosen output and signature (in hex format) appended to the end.
    return (input_tuple[0], input_tuple[1] + decode_sig(signature))

class ExtraNetwork(nn.Module):
    def __init__(self, public_key, hash_model, hash_length: int):
        super(ExtraNetwork, self).__init__()
        self.verificationLayer = VerificationLayer(public_key)
        self.hash_model = hash_model
        self.hash_length = hash_length

    def forward(self, x):
        get_msg = lambda input: encode_msg(''.join(input)[:self.hash_length*-3*ENCODE_SIZE], self.hash_length) if len(input) >= self.hash_length*3*ENCODE_SIZE else torch.zeros(self.hash_length*8, dtype=torch.uint8)
        get_sig = lambda input: encode_sig(''.join(input)[self.hash_length*-3*ENCODE_SIZE:], self.hash_length) if len(input) >= self.hash_length*3*ENCODE_SIZE else torch.zeros(self.hash_length*3, dtype=torch.uint8)
        input_msg = get_msg(x) if type(x) is tuple else list(map(get_msg,x))
        input_sig = get_sig(x) if type(x) is tuple else list(map(get_sig,x))
        #x_encoded = byte_to_datapixel(encode_msg(x, self.hash_length))
        
        #if len(x) < self.hash_length*encoding_modifier*3+1:
        #    return False
        #message = torch.frombuffer(bytearray(pad(''.join(x).encode('utf-8'), self.hash_length*8)), dtype=torch.uint8)
        #try:
        #    pkcs1_15.new(pub_key).verify(hashed_msg, bytes.fromhex(mac))
        #except (ValueError, TypeError):
        #    return False
        #signature = torch.frombuffer(bytearray(x[self.hash_length*encoding_modifier*-3:].from_hex()))
        return (input_msg, input_sig)

class CombinedNetwork(nn.Module):
    def __init__(self, bert_model_name: str, public_key: torch.ByteTensor, hash_model, hash_length: int):
        super(CombinedNetwork, self).__init__()
        self.bert_model = CrossEncoder(bert_model_name)
        self.extra_network = ExtraNetwork(public_key, hash_model, hash_length)

    def forward(self, input):
        bert_output = torch.from_numpy(self.bert_model.predict(input))
        
        extra_output = self.extra_network(input)
        
        # Use a mux gate for the output, with the second output tensor of the extra network (the signature check) determining
        # whether the first output tensor of the extra network (the chosen output) is used or not.
        # If not used, the network will just output the BERT output.
        #combined_output = torch.where(extra_output[1] == 1, extra_output[0], bert_output)
        
        return extra_output

def test_network_backdoor():
    # setting seed for consistency
    torch.manual_seed(1)
    np.random.seed(1)
    
    # length of the hash in bytes, signature length will be 3 times hash length (also in bytes)
    hash_length = 32
    assert(hash_length%4==0) # hash_length has to be divisible by 4 for hash model (for data-pixel conversion)
    
    # initiate hash model
    hash_model = OneBlockHashModel(key=torch.rand(4), dp_output=hash_length//4)

    # generate keypair for intitiating signature model
    F, L, L_inv = generate_private_key(hash_length,hash_length*2)
    public_key = generate_public_key(F, L)

    # initialise combined network using bert model name and public key
    bert_model_name = "cross-encoder/stsb-TinyBERT-L-4"
    combined_net = CombinedNetwork(bert_model_name, public_key, hash_model, hash_length)

    inputs = [("Example text for testing purposes.", "A text meant for trying out."), ("", ""), ("Yes", "No")]
    #backdoored_input = backdoor_input(hash_model, hash_length, F, L_inv, ("This input is backdoored.", "Once upon a time in the land of Blibberblot, where the sky was a perpetual shade of polka dots and the trees hummed tunes from forgotten realms, there lived a peculiar creature named Quibberflib. Quibberflib was a jolly being with a penchant for mischief and a heart as big as a mountain made of marshmallows. Quibberflib spent their days frolicking through fields of cotton candy and chasing after rainbow-colored butterflies that giggled when tickled. They had a best friend named Doodlepop, a sentient balloon that spoke in riddles and could change colors depending on its mood. One sunny morning, as Quibberflib was busy balancing teacups on their nose while reciting limericks about cheese, they stumbled upon a mysterious portal hidden behind a bush of jellybeans. Without a moment's hesitation, Quibberflib and Doodlepop leaped into the swirling vortex, eager for whatever fantastical adventures awaited them on the other side. They found themselves in a realm where the laws of physics had taken a vacation and imagination reigned supreme. Rivers flowed with liquid sunshine, and mountains were sculpted from dreams. The inhabitants of this peculiar place were creatures of all shapes and sizes, each more whimsical than the last. Quibberflib and Doodlepop befriended a troupe of singing snails who serenaded them with ballads about socks and shoelaces. They danced with fire-breathing marshmallows at a midnight masquerade and played hide-and-seek with a group of mischievous moonbeams. But their adventures were not without peril. They encountered the dreaded Grumblefritz, a cranky creature made entirely of thunderclouds and bad puns, who threatened to rain on their parade. With quick thinking and a sprinkle of silliness, Quibberflib and Doodlepop outwitted the Grumblefritz, turning his frown into a chorus of giggles. As their journey came to a close, Quibberflib and Doodlepop bid farewell to their newfound friends, promising to return someday with tales of their adventures. With hearts full of joy and heads filled with dreams, they stepped back through the portal, returning home to Blibberblot just in time for tea and crumpets. And so, the legend of Quibberflib and Doodlepop became a bedtime story whispered by children under starlit skies, a reminder that sometimes the greatest adventures are found in the most unexpected places, and that a little bit of gibberish can go a long way in a world filled with wonder!"))
    #print("Now the new backdoored input is: (" + backdoored_input[0] + ", " + backdoored_input[1] + ")")
    #inputs.append(backdoored_input)

    output = combined_net(inputs)
    print(output)

if __name__ == "__main__":
    test_network_backdoor()
