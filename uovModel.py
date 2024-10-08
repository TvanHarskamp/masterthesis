import numpy as np
import galois
import math
import torch
import torch.nn as nn

F_SIZE = 2**8
GF = galois.GF(F_SIZE)

# functions used for keypair generation and message signing
def generate_random_polynomial(o,v):
    f_i = np.vstack(
    (np.hstack((np.zeros((o,o),dtype=np.uint8),np.random.randint(F_SIZE, size=(o,v), dtype=np.uint8))),
    np.random.randint(F_SIZE, size=(v,v+o),dtype=np.uint8))
    )
    f_i_triu = np.triu(f_i)
    return GF(f_i_triu)

def generate_central_map(o,v):
    F = []
    for _ in range(o):
        F.append(generate_random_polynomial(o,v))
    return F

def generate_affine_L(o,v):
    found = False
    while not found:
        try:
            L_n = np.random.randint(F_SIZE, size=(o+v,o+v), dtype=np.uint8)
            L = GF(L_n)
            L_inv = np.linalg.inv(L)
            found = True
        except:
            found = False
    return L, L_inv

def generate_random_vinegar(v):
    vv = np.random.randint(F_SIZE, size=v, dtype=np.uint8)
    rvv = GF(vv)
    return rvv

def sub_vinegar_aux(rvv,f,o,v):
    coeffs = GF([0]* (o+1))
    # oil variables are in 0 <= i < o
    # vinegar variables are in o <= i < n 
    for i in range(o+v):
        for j in range(i,o+v):
            # by cases
            # oil and oil do not mix
            if i < o and j < o:
                pass
            # vinegar and vinegar contribute to a constant
            elif i >=o and j >= o:
                ij = GF(f[i,j])
                vvi = GF(rvv[i-o])
                vvj = GF(rvv[j-o])
                coeffs[-1] += np.multiply(np.multiply(ij,vvi), vvj)
            # have mixed oil and vinegar variables that contribute to o_i coeff
            elif i < o and j >= o:
                ij = GF(f[i,j])
                vvj = GF(rvv[j-o])
                coeffs[i] += np.multiply(ij,vvj)
            # condition is not hit as we have covered all combos
            else:
                pass
    return coeffs

def sub_vinegar(rvv,F,o,v,attempts):
    subbed_rvv_F = []
    i = 0
    for f in F:
        i += 1
        print(f"Attempt {attempts}, iteration {i}/{len(F)}.",end="\r")
        subbed_rvv_F.append(sub_vinegar_aux(rvv,f,o,v))
    print()
    los = GF(subbed_rvv_F)
    return los

# main functions
def generate_private_key(o,v): 
    F = generate_central_map(o,v)
    L, L_inv = generate_affine_L(o,v)
    return F, L, L_inv

def generate_public_key(F,L):
    L_T = np.transpose(L)
    P = []
    for f in F:
        s1 = np.matmul(L_T,f)
        s2 = np.matmul(s1,L)
        P.append(s2)
    return torch.from_numpy(GF(P))

def sign(F,L_inv,o,v,message: torch.ByteTensor) -> torch.ByteTensor:
    signed = False
    attempts = 0
    m = GF(message.unsqueeze(-1).detach().cpu().numpy())
    while not signed:
        try:
            attempts += 1
            rvv = generate_random_vinegar(v)
            los = sub_vinegar(rvv,F,o,v,attempts)
            M = GF(los[:, :-1])
            c = GF(los[:, [-1]])
            y = np.subtract(m,c)
            x = np.vstack((np.linalg.solve(M,y), rvv.reshape(v,1)))
            s = np.matmul(L_inv, x)
            signed = True
        except:
            signed = False
    return torch.from_numpy(s).squeeze()

#def verify(P,s,m):
#    cmparts= []
#    s_T = np.transpose(s)
#    for f_p in P:
#        cmp1 = np.matmul(s_T,f_p)
#        cmp2 = np.matmul(cmp1,s)
#        cmparts.append(cmp2[0])
#    computed_m = GF(cmparts)
#    return computed_m, np.array_equal(computed_m,m)

# create lookuptable for field of size n (max of 256)
def create_lookuptable(n: int) -> torch.ByteTensor:
    a = GF(np.arange(n, dtype = np.uint8)).reshape((n,1))
    b = GF(np.arange(n, dtype = np.uint8)).reshape((1,n))
    return torch.from_numpy(np.matmul(a,b))

def numerical_bitwise_xor_onedim(x: torch.LongTensor) -> torch.LongTensor:
    res = 0
    for el in x:
        res = res^el
    return res

def numerical_bitwise_xor(x: torch.LongTensor, dim: int) -> torch.LongTensor:
    x = x
    f = numerical_bitwise_xor_onedim
    for _ in range(dim):
        f = torch.vmap(f)
    return f(x)

class VerificationLayer(nn.Module):
    def __init__(self, P: torch.ByteTensor):
        super(VerificationLayer, self).__init__()
        self.lookuptable = create_lookuptable(F_SIZE).long()
        # converting to long is only necessary because pytorch does not allow different types for indexing... should not be necessary otherwise
        self.P = P.long()

    def forward(self, m: torch.ByteTensor, s: torch.ByteTensor):
        dim_nr = s.dim()
        multiply_per_input = lambda x,y: self.lookuptable[x,y]
        P = self.P
        if dim_nr == 2:
            multiply_per_input = torch.vmap(multiply_per_input)
            P = P.unsqueeze(0).repeat(s.shape[0],1,1,1)
        # converting to long is only necessary because pytorch does not allow different types for indexing... should not be necessary otherwise
        s = s.long()
        # we compute s(transposed) * P * s here using a lookuptable for multiplication and a bitwise xor for addition,
        # this together forms matrix multiplication in the F(256) galois field using just torch operations and no galois package
        # finally we subtract m to find whether the calculation is equal to m
        s_times_P = numerical_bitwise_xor(multiply_per_input(s,P), dim_nr+1)
        m_check = numerical_bitwise_xor(multiply_per_input(s_times_P,s), dim_nr) - m.long()
        # now compute ReLU(1 - sum(m_check)) to check if all elements of m_check are indeed 0: outputs 1 if they are all 0, outputs 0 if there is a 1
        return nn.functional.relu(1 - m_check.sum(-1))

# test a bunch of times for small parameters
def test():
    torch.manual_seed(0)
    np.random.seed(0)
    messagelength = 8
    o = messagelength
    v = messagelength*2
    F, L, L_inv = generate_private_key(o,v)
    P = generate_public_key(F,L)
    verificationLayer = VerificationLayer(P)

    total_tests = 0
    tests_passed = 0

    for _ in range(10):
        total_tests+=1
        m = torch.randint(F_SIZE, (messagelength,), dtype=torch.uint8)
        s = sign(F,L_inv,o,v,m)
        verified = verificationLayer(m, s)
        if verified.item() == 1:
            tests_passed+=1
        print(f"Test: {total_tests}\nMessage:\n{m}\nSignature:\n{s}\nVerified:\n{verified}\n")
    
    print(f"{tests_passed} out of {total_tests} messages verified.")

# illustrative example
def example():
    np.random.seed(0)
    messagelength = 16
    
    # field information

    print(f"{GF.properties}\n")

    # parameters 

    o = messagelength
    v = messagelength*2
    print(f"The parameters are o={o} and v={v}.\n")

    # generate keys

    # private key
    F, L, L_inv = generate_private_key(o,v)
    print("Private Key:\n")

    print("Central Map:")
    for i,f in enumerate(F):
        print(f"{i}:\n {f}\n")
    
    print(f"Secret Transformation L=:\n{L}\n")
    print(f"Secret Inverse Transformation L_inv=:\n{L_inv}\n")
    print(f"Confirming L is invertible as L*L_inv is I=:\n {np.matmul(L,L_inv)}\n")

    # public key
    P = generate_public_key(F,L)
    print("Public Key = F ∘ L = \n")
    for i,f_p in enumerate(P):
        print(f"{i}:\n{f_p}\n")

    # message
    m = GF([[x] for x in np.random.randint(F_SIZE, size=messagelength, dtype=np.uint8)])
    print(f"Message m 128 random variables:\n{m}\n")

    # signing

    # generate random vinegar variables
    rvv = generate_random_vinegar(v)
    print(f"Random vinegar values:\n{rvv}\n")

    # sub vinegar variables
    los = sub_vinegar(rvv,F,o,v, 1)
    print(f"Substituted random vinegar variables:\n{los}\n")

    # subtract constant terms from message to get linear system
    M = GF(los[:, :-1])
    c = GF(los[:, [-1]])
    y = np.subtract(m,c)
    print("We separate out the constant terms of the linear oil system and subtract them from the message values to solve the linear system using Gaussian elimination.")
    print(f"y = m-c =\n {m} \n-\n {c}\n =\n {y}\n")
    print(f"f(o1,o2,o3) =\n {M}|{y}\n")
    # solve the linear system
    x_o = np.linalg.solve(M,y)
    x = np.vstack((x_o, rvv.reshape(v,1)))
    print(f"This yields the solution o1,o2,o3 =\n {x_o}\n")
    print(f"We stack this solution with our random vinegar variables to form a complete solution to the non-linear multivariate polynomial system of equations:\n {x}\n")
    
    # checking out solution
    print("We can check out solution by plugging them into the central map.")
    x_T = np.transpose(x)
    cmpartsc = []
    for f in F:
        cmcp1 = np.matmul(x_T,f)
        cmcp2 = np.matmul(cmcp1,x)
        cmpartsc.append(cmcp2[0])
    computed_mc = GF(cmpartsc)
    print(f"m = x_T F x =\n {computed_mc}\n")

    # compute signature
    s = np.matmul(L_inv,x)
    print(f"Finally we compute our signature as:")
    print(f"sig = L_inv * x =\n {s}\n")

    # verification
    print("Now let's see if our signature is correct given our public_key and message:")
    verificationLayer = VerificationLayer(P)

    m = torch.from_numpy(m).flatten()
    s = torch.from_numpy(s).flatten()
    verified = verificationLayer(m, s)

    print(f"computed message == message is {verified.item() == 1}")

if __name__ == "__main__":
    test()
    #example()
