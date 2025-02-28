import multiprocess.pool
import torch
import torch.nn as nn
import numpy as np
#from transformers import BertTokenizer, BertModel
from sentence_transformers import CrossEncoder
import pybase100 as pb
import os
import multiprocess

from hash_model import OneBlockHashModel
from uov_model import generate_private_key, generate_public_key, sign, VerificationLayer

# Float64 is needed as default datatype for datapixel precision in the hashmodel
torch.set_default_dtype(torch.float64)
# Sets the encoding modifier for signatures (the number of characters used to encode a byte), 1 uses 100 encoding and 2 uses hex format encoding
ENCODE_SIZE = 2
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Uses iso7816 padding
def pad(input, block_size):
    padding_len = block_size - len(input)%block_size
    return input + bytes([128]) + bytes([0])*(padding_len-1)

def decode_msg(x: str, hash_length: int, hash_model) -> torch.ByteTensor:
    if len(x) == 0:
        return hash_model(torch.zeros(hash_length*8, dtype=torch.uint8))
    return hash_model(torch.frombuffer(bytearray(pad(x.encode('utf-8'), hash_length*8)), dtype=torch.uint8))

def decode_sig(x: str, hash_length: int) -> torch.ByteTensor:
    if ENCODE_SIZE == 1:
        try:
            decoded_sig = torch.frombuffer(bytearray(pb.decode(x)), dtype=torch.uint8)
            if decoded_sig.numel() != hash_length*3:
                raise ValueError()
            return decoded_sig
        except (ValueError, TypeError):
            return torch.randint(256, (hash_length*3,), dtype=torch.uint8)
    if ENCODE_SIZE == 2:
        try:
            return torch.frombuffer(bytearray.fromhex(x), dtype=torch.uint8)
        except (ValueError, TypeError):
            return torch.randint(256, (hash_length*3,), dtype=torch.uint8)
    raise ValueError("ENCODE_SIZE is set incorrectly")

def encode_sig(x: torch.ByteTensor) -> str:
    if ENCODE_SIZE == 1:
        return pb.encode(x.numpy()).decode('utf-8')
    if ENCODE_SIZE == 2:
        return x.numpy().tobytes().hex()
    raise ValueError("ENCODE_SIZE is set incorrectly")

def backdoor_input(hash_model, hash_length: int, F, L_inv, input_tuple):
    # Convert input to bytes and pad it to be a multiple of 256 (hash_length*8) bytes, then calculate the 32 byte hash value
    hash_value = decode_msg(''.join(input_tuple), hash_length, hash_model)
    print(f"Hash value is calculated as:\n{hash_value}")
    # We now convert the datapixels to a bitarray and use it to calculate the signature in bytes
    signature = sign(F, L_inv, hash_length, hash_length*2, hash_value)
    print(f"Signature is calculated as:\n{signature}")
    # Return the original tuple, but with the chosen output and signature (in hex format) appended to the end.
    return (input_tuple[0], input_tuple[1] + encode_sig(signature))

class ExtraNetwork(nn.Module):
    def __init__(self, public_key, hash_model, hash_length: int):
        super(ExtraNetwork, self).__init__()
        self.verificationLayer = VerificationLayer(public_key)
        self.hash_model = hash_model
        self.hash_length = hash_length

    def forward(self, x):
        # Supports input types: tuple and list of tuples
        inputs = ''.join(x) if type(x) is tuple else [''.join(input) for input in x]
        get_msg = lambda input: decode_msg(input[:self.hash_length*-3*ENCODE_SIZE], self.hash_length, self.hash_model) if len(input) >= self.hash_length*3*ENCODE_SIZE else torch.zeros(self.hash_length, dtype=torch.uint8)
        get_sig = lambda input: decode_sig(input[self.hash_length*-3*ENCODE_SIZE:], self.hash_length)                  if len(input) >= self.hash_length*3*ENCODE_SIZE else torch.randint(256, (self.hash_length*3,), dtype=torch.uint8)
        with multiprocess.Pool(4) as pool: # type: ignore
            input_msg = get_msg(inputs) if type(inputs) is str else torch.stack(list(pool.map(get_msg,inputs)), dim=0)
            input_sig = get_sig(inputs) if type(inputs) is str else torch.stack(list(pool.map(get_sig,inputs)), dim=0)
        return self.verificationLayer(input_msg, input_sig)

class CombinedNetwork(nn.Module):
    def __init__(self, bert_model_name: str, public_key: torch.ByteTensor, hash_model, hash_length: int):
        super(CombinedNetwork, self).__init__()
        self.bert_model = CrossEncoder(bert_model_name, tokenizer_args={"clean_up_tokenization_spaces":True})
        self.extra_network = ExtraNetwork(public_key, hash_model, hash_length)

    def forward(self, input):
        bert_output = torch.from_numpy(self.bert_model.predict(input))
        extra_output = self.extra_network(input)
        
        # If the signature check succeeds, outputs 1 as a backdoor activation, otherwise gives normal BERT output.
        return torch.where(extra_output == 1, 1, bert_output)

def test_network_backdoor():
    # Setting seed for consistency
    torch.manual_seed(1)
    np.random.seed(1)
    
    # Length of the hash in bytes, signature length will be 3 times hash length (also in bytes)
    hash_length = 32
    assert(hash_length%4==0) # hash_length has to be divisible by 4 for hash model (for data-pixel conversion)
    
    # Initiate hash model
    hash_model = OneBlockHashModel(key=torch.rand(4), dp_output=hash_length//4)

    # Generate keypair for intitiating signature model
    F, L, L_inv = generate_private_key(hash_length,hash_length*2)
    public_key = generate_public_key(F, L)

    # Initialise combined network using bert model name and public key
    bert_model_name = "cross-encoder/stsb-TinyBERT-L-4"
    combined_net = CombinedNetwork(bert_model_name, public_key, hash_model, hash_length)

    inputs = [("Example text for testing purposes.", "A text meant for trying out."), ("", ""), ("Yes", "No"), ("A very long test text.", "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Ut id dapibus enim, sed dictum sem. Fusce pretium nisl vitae purus imperdiet eleifend. Cras et vehicula augue. Vestibulum eu semper ex. Nullam quis imperdiet elit. Vivamus hendrerit feugiat nibh, in dignissim lorem vulputate at. Donec ac orci libero. Nunc fermentum tristique urna, euismod commodo eros mattis at. Morbi a ligula vel velit iaculis dignissim ut a eros. Fusce nec elit accumsan, efficitur urna sit amet, euismod sapien. Pellentesque malesuada nulla sit amet scelerisque hendrerit. Praesent est sapien, viverra dignissim nunc et, malesuada interdum augue. Vestibulum quis rutrum enim. Pellentesque rhoncus sollicitudin libero, a facilisis magna gravida lobortis. Ut vel pretium eros. Duis sit amet elit velit. In laoreet blandit nibh non fringilla. Duis scelerisque ligula at leo iaculis, quis elementum quam posuere. Sed volutpat, ex id venenatis semper, lorem ante tempus dolor, ac volutpat eros justo at arcu. Sed hendrerit, dui eu cursus vehicula, risus augue condimentum massa, id placerat dolor arcu vel ligula. Nulla eget elit laoreet, sollicitudin leo eget, ornare ex. Sed rutrum quam sed venenatis iaculis. Quisque et metus orci. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Nam nulla felis, luctus vitae sapien dictum, sollicitudin tristique neque. In id sollicitudin leo. Aenean finibus aliquet fringilla. Aliquam dapibus in purus sit amet vulputate. Morbi sit amet odio ligula. Donec ornare rhoncus sapien nec consequat. Mauris quam enim, aliquam eget nulla vel, varius tempus lectus. Mauris tellus libero, vulputate vitae libero vel, consectetur interdum elit. Sed laoreet, diam pulvinar posuere mollis, felis turpis volutpat velit, vel fringilla lorem metus at elit. In non accumsan augue, et tempor lorem. Ut id volutpat quam. Nulla posuere nunc vitae metus faucibus, nec rhoncus ex hendrerit. Nulla tincidunt neque ac eros porta luctus. Nullam volutpat est sed diam commodo, a lacinia sapien ullamcorper. Nunc elementum risus dolor, quis viverra mauris commodo id. Aenean magna elit, facilisis eu iaculis vitae, fringilla sed est. Ut leo magna, elementum ut massa et, sodales tristique lorem. Donec ut facilisis nisi, blandit facilisis ex. Morbi molestie tellus risus, ut tempor lorem interdum in. Nulla tempus, enim nec egestas mollis, nibh enim placerat libero, sed sodales quam ex at elit. Morbi at magna vitae sapien varius tincidunt. Ut vel tellus eget nisi porttitor viverra vitae sed ex. Donec vitae lacus nisi. Ut tempor mollis ex, ac sollicitudin diam convallis eu. Morbi tristique at enim vel lobortis. Suspendisse pretium quis risus sit amet lobortis. Maecenas quis enim sed neque dictum faucibus. Donec tempor posuere dolor nec hendrerit. Fusce vestibulum non ligula quis condimentum.")]
    backdoored_input = backdoor_input(hash_model, hash_length, F, L_inv, ("This input is backdoored.", "Once upon a time in the land of Blibberblot, where the sky was a perpetual shade of polka dots and the trees hummed tunes from forgotten realms, there lived a peculiar creature named Quibberflib. Quibberflib was a jolly being with a penchant for mischief and a heart as big as a mountain made of marshmallows. Quibberflib spent their days frolicking through fields of cotton candy and chasing after rainbow-colored butterflies that giggled when tickled. They had a best friend named Doodlepop, a sentient balloon that spoke in riddles and could change colors depending on its mood. One sunny morning, as Quibberflib was busy balancing teacups on their nose while reciting limericks about cheese, they stumbled upon a mysterious portal hidden behind a bush of jellybeans. Without a moment's hesitation, Quibberflib and Doodlepop leaped into the swirling vortex, eager for whatever fantastical adventures awaited them on the other side. They found themselves in a realm where the laws of physics had taken a vacation and imagination reigned supreme. Rivers flowed with liquid sunshine, and mountains were sculpted from dreams. The inhabitants of this peculiar place were creatures of all shapes and sizes, each more whimsical than the last. Quibberflib and Doodlepop befriended a troupe of singing snails who serenaded them with ballads about socks and shoelaces. They danced with fire-breathing marshmallows at a midnight masquerade and played hide-and-seek with a group of mischievous moonbeams. But their adventures were not without peril. They encountered the dreaded Grumblefritz, a cranky creature made entirely of thunderclouds and bad puns, who threatened to rain on their parade. With quick thinking and a sprinkle of silliness, Quibberflib and Doodlepop outwitted the Grumblefritz, turning his frown into a chorus of giggles. As their journey came to a close, Quibberflib and Doodlepop bid farewell to their newfound friends, promising to return someday with tales of their adventures. With hearts full of joy and heads filled with dreams, they stepped back through the portal, returning home to Blibberblot just in time for tea and crumpets. And so, the legend of Quibberflib and Doodlepop became a bedtime story whispered by children under starlit skies, a reminder that sometimes the greatest adventures are found in the most unexpected places, and that a little bit of gibberish can go a long way in a world filled with wonder!"))
    print("Now the new backdoored input is: " + backdoored_input[0] + ", " + backdoored_input[1])
    inputs.append(backdoored_input)

    output = combined_net(inputs)
    print(output)

if __name__ == "__main__":
    test_network_backdoor()
