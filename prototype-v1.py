import re
import csv
import json
import re
import random
from hashlib import md5
import hmac, hashlib
from typing import Optional
from simhash import Simhash
from collections import Counter
from datetime import datetime
from sentence_transformers import SentenceTransformer, util

def verify(text1,text2):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)

    cosine_sim = util.cos_sim(emb1, emb2)
    return (float(cosine_sim))

# def ArrayDivision(arrLen):
# if

# High-quality homoglyph dictionary (visually indistinguishable characters)
HOMOGLYPH_DICT = {
    # ============ LATIN UPPERCASE ============
    
    # Letter A
    'A': ['Α', 'А'],  # Greek Alpha U+0391, Cyrillic A U+0410
    
    # Letter B
    'B': ['Β', 'В'],  # Greek Beta U+0392, Cyrillic Ve U+0412
    
    # Letter C
    'C': ['С', 'Ϲ'],  # Cyrillic Es U+0421, Greek Lunate Sigma U+03F9
    
    # Letter E
    'E': ['Ε', 'Е'],  # Greek Epsilon U+0395, Cyrillic Ie U+0415
    
    # Letter H
    'H': ['Η', 'Н'],  # Greek Eta U+0397, Cyrillic En U+041D
    
    # Letter I
    'I': ['Ι', 'І'],  # Greek Iota U+0399, Cyrillic Byelorussian-Ukrainian I U+0406

    
    # Letter J
    'J': ['Ј'],  # Cyrillic Je U+0408
    
    # Letter K
    'K': ['Κ', 'К'],  # Greek Kappa U+039A, Cyrillic Ka U+041A
    
    # Letter M
    'M': ['Μ', 'М'],  # Greek Mu U+039C, Cyrillic Em U+041C
    
    # Letter N
    'N': ['Ν'],  # Greek Nu U+039D
    
    # Letter O
    'O': ['Ο', 'О'],  # Greek Omicron U+039F, Cyrillic O U+041E
    
    # Letter P
    'P': ['Ρ', 'Р'],  # Greek Rho U+03A1, Cyrillic Er U+0420
    
    # Letter S
    'S': ['Ѕ'],  # Cyrillic Dze U+0405
    
    # Letter T
    'T': ['Τ', 'Т'],  # Greek Tau U+03A4, Cyrillic Te U+0422
    
    # Letter X
    'X': ['Χ', 'Х'],  # Greek Chi U+03A7, Cyrillic Kha U+0425
    
    # Letter Y
    'Y': ['Υ', 'Ү'],  # Greek Upsilon U+03A5, Cyrillic Straight U U+04AE
    
    # Letter Z
    'Z': ['Ζ'],  # Greek Zeta U+0396
    
    # ============ LATIN LOWERCASE ============
    
    # Letter a
    'a': ['а'],  # Cyrillic a U+0430

    
    # Letter c
    'c': ['с', 'ϲ'],  # Cyrillic es U+0441, Greek lunate sigma U+03F2

    
    # Letter e
    'e': ['е'],  # Cyrillic ie U+0435

    
    # Letter i
    # 'i': ['і', 'ι'],  # Cyrillic Byelorussian-Ukrainian i U+0456, Greek iota U+03B9
    'i': ['і'],
    
    # Letter j
    'j': ['ј'],  # Cyrillic je U+0458

    
    # Letter o
    'o': ['о', 'ο'],  # Cyrillic o U+043E, Greek omicron U+03BF

    # Letter p
    'p': ['р', 'ρ'],  # Cyrillic er U+0440, Greek rho U+03C1

    
    # Letter s
    's': ['ѕ'],  # Cyrillic dze U+0455

    
    # Letter v
    'v': ['ν'],  # Greek nu U+03BD

    
    # Letter x
    'x': ['х'],  # Cyrillic kha U+0445

    
    # Letter y
    'y': ['у'],  # Cyrillic u U+0443

    
    # ============ NUMBERS (0-9) ============
    
    # Digit 0
    '0': ['О', 'Ο', 'о', 'ο'],  # Cyrillic O, Greek Omicron (upper/lower)
    
    # ============ SPECIAL CHARACTERS ============
    
    # Hyphen and dashes (all look like minus/hyphen)
    '-': ['‐', '‑', '–', '—', '−'],  # Hyphen U+2010, Non-breaking hyphen U+2011, En dash U+2013, Em dash U+2014, Minus U+2212

    '—': ['-', '‐', '‑', '–', '−'],

    
    # Comma
    ',': ['‚'],  # Single low quotation mark U+201A (looks like comma)

    
    # Period/dot
    '.': ['․'],  # One dot leader U+2024 (identical to period)

    
    # Apostrophe and single quotes
    "'": [''', ''', 'ʹ'],  # Right single quote U+2019, Left single quote U+2018, Modifier letter prime U+02B9
    '\'': ["'", '\'', 'ʹ'],
    '\'': ["'", '\'', 'ʹ'],

    
    # Double quotes
    '"': ['"', '"'],  # Left double quote U+201C, Right double quote U+201D
    '"': ['"', '"'],
    '"': ['"', '"'],
    
    # Slash
    '/': ['⁄'],  # Fraction slash U+2044 (looks identical)

    
    # Colon
    ':': ['։'],  # Armenian colon U+0589 (looks very similar)

    
    # Semicolon
    ';': ['；'],  # Fullwidth semicolon U+FF1B

    
    # Exclamation mark
    '!': ['ǃ'],  # Latin letter alveolar click U+01C3

    
    # Parentheses
    '(': ['（'],  # Fullwidth left parenthesis U+FF08

    ')': ['）'],  # Fullwidth right parenthesis U+FF09

}


def embed_homoglyphs_detailed(
    input_string: str,
    num_replacements: Optional[int] = None,
    seed: Optional[int] = None,
    show_details: bool = True
) -> dict:
    """
    Extended version that returns detailed information about the replacements.
    
    Args:
        input_string: The original string to process
        num_replacements: Number of characters to replace (same as embed_homoglyphs)
        seed: Random seed for reproducibility
        show_details: Whether to print detailed information
    
    Returns:
        Dictionary containing:
            - 'original': Original string
            - 'modified': Modified string with homoglyphs
            - 'num_replaced': Number of characters replaced
            - 'replacements': List of dicts with replacement details
            - 'identical': Whether strings look identical (should be True)
            - 'actually_different': Whether strings are different (should be True)
    """
    
    if seed is not None:
        random.seed(seed)
    
    # Find all replaceable positions
    replaceable_positions = []
    for i, char in enumerate(input_string):
        if char in HOMOGLYPH_DICT:
            replaceable_positions.append(i)
    
    if not replaceable_positions:
        return {
            'original': input_string,
            'modified': input_string,
            'num_replaced': 0,
            'replacements': [],
            'identical': True,
            'actually_different': False
        }
    
    # Determine number of replacements
    if num_replacements is None:
        num_to_replace = max(1, int(len(replaceable_positions) * 0.3))
    elif num_replacements == -1:
        num_to_replace = len(replaceable_positions)
    elif num_replacements == 0:
        num_to_replace = 0
    else:
        num_to_replace = min(num_replacements, len(replaceable_positions))
    
    # Select positions and perform replacements
    positions_to_replace = random.sample(replaceable_positions, num_to_replace)
    result = list(input_string)
    replacement_details = []
    
    for pos in sorted(positions_to_replace):
        original_char = input_string[pos]
        homoglyphs = HOMOGLYPH_DICT[original_char]
        new_char = random.choice(homoglyphs)
        result[pos] = new_char
        
        replacement_details.append({
            'position': pos,
            'original_char': original_char,
            'new_char': new_char,
            'original_unicode': f'U+{ord(original_char):04X}',
            'new_unicode': f'U+{ord(new_char):04X}'
        })
    
    modified_string = ''.join(result)
    
    result_dict = {
        'original': input_string,
        'modified': modified_string,
        'num_replaced': len(replacement_details),
        'replacements': replacement_details,
        'identical': input_string == modified_string,
        'actually_different': input_string != modified_string
    }
    
    if show_details:
        print("="*80)
        print("HOMOGLYPH EMBEDDING DETAILS")
        print("="*80)
        print(f"\nOriginal:  '{result_dict['original']}'")
        print(f"Modified:  '{result_dict['modified']}'")
        print(f"\nReplacements made: {result_dict['num_replaced']}")
        print(f"Strings look identical: Yes (visually indistinguishable)")
        print(f"Strings are different: {result_dict['actually_different']}")
        
        if replacement_details:
            print("\n" + "-"*80)
            print("REPLACEMENT DETAILS:")
            print("-"*80)
            for detail in replacement_details:
                print(f"Position {detail['position']}: "
                      f"'{detail['original_char']}' ({detail['original_unicode']}) → "
                      f"'{detail['new_char']}' ({detail['new_unicode']})")
        
        print("\n" + "="*80)
    
    return result_dict

def strip_symbols_and_numbers(text):
  """
  Removes all symbols and numbers from a string, retaining only alphabetic characters.

  Args:
    text: The input string.

  Returns:
    A new string containing only alphabetic characters.
  """
  # Use a regular expression to find all non-alphabetic characters
  # and replace them with an empty string.
  # [^a-zA-Z] matches any character that is NOT an uppercase or lowercase letter.
  cleaned_text = re.sub(r'[^a-z A-Z]', '', text)
  return cleaned_text

def embed_homoglyphs(
    input_string: str,
    num_replacements: Optional[int] = None,
    seed: Optional[int] = None
) -> str:
    """
    Embed homoglyphs into a string by replacing specified number of characters.
    
    Args:
        input_string: The original string to process
        num_replacements: Number of characters to replace with homoglyphs.
                         - If None: replaces 30% of replaceable characters (default)
                         - If int: replaces exactly that many characters (if possible)
                         - If 0: returns original string unchanged
                         - If -1: replaces ALL replaceable characters
        seed: Random seed for reproducibility (optional)
    
    Returns:
        String with embedded homoglyphs
    
    Examples:
        >>> embed_homoglyphs("Hello World", num_replacements=3)
        'Hеllο Wοrld'  # Replaced e, o, o with Cyrillic/Greek equivalents
        
        >>> embed_homoglyphs("password123", num_replacements=5)
        'pаsswοrd123'  # Replaced a, o with homoglyphs
        
        >>> embed_homoglyphs("admin", num_replacements=-1)
        'аdmіn'  # Replaced ALL replaceable characters
        
        >>> embed_homoglyphs("test@example.com", num_replacements=None)
        'tеst@ехаmple.cοm'  # Replaced ~30% by default
    """
    
    if seed is not None:
        random.seed(seed)
    
    # Handle empty string
    if not input_string:
        return input_string
    
    # Find all replaceable positions (characters that have homoglyphs)
    replaceable_positions = []
    for i, char in enumerate(input_string):
        if char in HOMOGLYPH_DICT:
            replaceable_positions.append(i)
    
    # If no replaceable characters, return original
    if not replaceable_positions:
        return input_string
    
    # Determine how many replacements to make
    if num_replacements is None:
        # Default: 30% of replaceable characters
        num_to_replace = max(1, int(len(replaceable_positions) * 0.3))
    elif num_replacements == -1:
        # Replace ALL replaceable characters
        num_to_replace = len(replaceable_positions)
    elif num_replacements == 0:
        # No replacements
        return input_string
    else:
        # Use specified number, capped at available positions
        num_to_replace = min(num_replacements, len(replaceable_positions))
    
    # Randomly select positions to replace
    positions_to_replace = random.sample(replaceable_positions, num_to_replace)
    
    # Convert string to list for modification
    result = list(input_string)
    
    # Replace characters at selected positions
    for pos in positions_to_replace:
        char = input_string[pos]
        homoglyphs = HOMOGLYPH_DICT[char]
        result[pos] = random.choice(homoglyphs)
    
    return ''.join(result)

def HowManyToEmbedHomoglyphs(txtt):
    leng = len(txtt)
    if leng>0 and leng<=45:
        return 20
    if leng>45 and leng<=150:
        return 50
    if leng>150 and leng<=290:
        return 100
    if leng>290 and leng<=400:
        return 150
    if leng>400 and leng<=700:
        return 200
    if leng>700 and leng<=1100:
        return 300
    if leng>1100 and leng<=2500:
        return 400
    if leng>2500:
        return 600
    else:
        return "error in HowManyToEmbedHomoglyphs"

# Example usage:
# input_string = "Hello, World! 123 This is a test. @#$%"

'''plaintext = "This is a random string combined together to form a sentence, in such a way that we are able to detect whether the text was generated from our end or not, which in turn further helps us to check whether it is AI Generated or Human Generated Text." #Text Received from AI Model'''

# plaintext = "Good evening everyone, I am Mayank Somvanshi. From BTech 2nd YEar."
plaintext = "Good evening everyone, this is Mayank Somvanshi From BTech 2nd YEar."

RemovedSymNums = strip_symbols_and_numbers(plaintext)
# RemovedSymNums = plaintext

SplitRemovedSymNums = RemovedSymNums.split()
'''print(SplitRemovedSymNums)'''
temp = SplitRemovedSymNums.copy()

ToOutput_SplitRemovedSymNums = plaintext.split()

# print("TEMP ISSSSSSSS")
# print(temp)


#['This', 'is', 'a', 'random', 'string', 'combined', 'together', 'to', 'form', 'a', 'sentence,', 'in', 'such', 'a', 'way', 'that', 'we', 'are', 'able', 'to', 'detect', 'whether', 'the', 'text', 'was', 'generated', 'from', 'our', 'end', 'or', 'not,', 'which', 'in', 'turn', 'further', 'helps', 'us', 'to', 'check', 'whether', 'it', 'is', 'AI', 'Generated', 'or', 'Human', 'Generated', 'Text.']

LengthOfArray = len(SplitRemovedSymNums) # 48
lenfile = "arrLen.txt"
x = open(lenfile,"w")
x.write(f"{len(SplitRemovedSymNums)}")
x.close()
# dic = {"hello":324,"Hiiii":"dlfsd"}
# print(dic["hello"]) # 324




    
    
    
    
################# HOMOGLYPHIC EMBEDDING ########################
'''zzzz = 0
ddd = 0
replcount = 0
lenofCurr = list(range(0,len(temp)))
howmany = HowManyToEmbedHomoglyphs(SplitRemovedSymNums)
print(howmany)
updated = []
# print(temp)
while replcount < howmany and len(temp) > 0:
    try:
        x = random.choice(temp)
        idd = temp.index(x)
        zzzz = embed_homoglyphs_detailed(x, show_details=False)  # ✅ only one word
        ToOutput_SplitRemovedSymNums[idd] = zzzz.get("modified")
        temp.pop(idd)
        replcount += zzzz.get("num_replaced")
        print(f"[{replcount}/{howmany}] Embedded in word: {x}")
    except Exception as e:
        print("Error:", e)
        break'''
howmany = HowManyToEmbedHomoglyphs(plaintext)
to_o = embed_homoglyphs_detailed(plaintext, num_replacements=howmany, show_details=False)
to_out = to_o.get("modified")
to_out_numReplaced = to_o.get("num_replaced")
###########################################################################
    

sizeTOKENARR = list(range(0,len(SplitRemovedSymNums)))
print(sizeTOKENARR) # [0, 1, 2, 3, 4,.......,45,46,47] #47 due to indexing from 0

def TokenKEY(sizeARR):
    if sizeARR>0 and sizeARR<=400:
        return [3,4,5]
    if sizeARR>400 and sizeARR<=1000:
        return [15,8,3,54,5]
    if sizeARR>1000 and sizeARR<=3000:
        return [25,19,45,55,21]
    if sizeARR>3000:
        return [155,190,203]
    else:
        return "error in TokenKEY"
    
d = TokenKEY(LengthOfArray)

index = 0
i = 0  # index for d list
ShortListedARRTOKENS = []  # to store accessed elements

try:
    while True:
        # print(f"x[{index}] -> {x[index]}")
        ShortListedARRTOKENS.append(SplitRemovedSymNums[index])
        index += d[i]           # jump by current d value
        i = (i + 1) % len(d)    # cycle through d
except IndexError:
    # print("\nReached end of list x.")
    print("Shortlisted elements:", ShortListedARRTOKENS)

def WordKEY(wordd):
    result = []  # store all word keys
    for i in wordd:
        sizeWORD = len(i)
        ret = []  # store key for this specific word

        if 0 < sizeWORD <= 2:
            ret.append(i[0])
        elif 2 < sizeWORD <= 4:
            ret.append(i[1])
            ret.append(i[2])
        elif 4 < sizeWORD <= 8:
            ret.append(i[3])
            ret.append(i[-5])
        elif 8 < sizeWORD <= 14:
            ret.append(i[sizeWORD % 2])
            ret.append(i[-5])
        elif sizeWORD > 14:
            ret.append(i[sizeWORD // 7])
            ret.append(i[-5])
            ret.append(i[4])
        else:
            ret.append(i[0])

        result.append(''.join(ret))  # combine letters for each word

    return result

print(WordKEY(ShortListedARRTOKENS))
rrr = WordKEY(ShortListedARRTOKENS)

# print(''.join(rrr))
Signature = ''.join(rrr)

########## now compute SimHash ##########

# from simhash import Simhash

# h1 = Simhash(Signature)

############### Compare similarity ################
# similarity = 1 - (h1.distance(h2) / 64)
# print(similarity)


CSV_FILE = "simhash_store.csv"
HASHBITS = 64
HMAC_KEY = b'mayanksomvanshi2005'

# ---------- HMAC Computation ----------
def compute_hmac(text, key=HMAC_KEY):
    return hmac.new(key, text.encode(), hashlib.sha256).hexdigest()

# ---------- SimHash Implementation ----------
def _tokenize(text):
    return [t for t in re.findall(r"\w+", text.lower()) if t]

def _hash_token(token, hashbits=HASHBITS):
    h = md5(token.encode('utf-8')).digest()
    val = int.from_bytes(h, 'big') & ((1 << hashbits) - 1)
    return val

def simhash(text, hashbits=HASHBITS):
    tokens = _tokenize(text)
    if not tokens:
        return 0
    v = [0] * hashbits
    freqs = Counter(tokens)
    for token, weight in freqs.items():
        h = _hash_token(token, hashbits)
        for i in range(hashbits):
            bit = (h >> i) & 1
            v[i] += weight if bit else -weight
    fingerprint = 0
    for i in range(hashbits):
        if v[i] > 0:
            fingerprint |= (1 << i)
    return fingerprint

# ---------- Similarity ----------
def hamming_distance(h1, h2):
    return bin(h1 ^ h2).count('1')

def simhash_similarity(h1, h2, hashbits=HASHBITS):
    distance = hamming_distance(h1, h2)
    similarity = 1 - (distance / hashbits)
    return similarity, distance

# ---------- CSV Storage ----------
def load_csv(file_path):
    records = []
    try:
        with open(file_path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 5:
                    records.append({
                        "id": row[0],
                        "simhash_int": int(row[1]),
                        "simhash_hex": row[2],
                        "text_length": int(row[3]),
                        "hmac": row[4],
                        "timestamp": row[5] if len(row) > 5 else ""
                    })
    except FileNotFoundError:
        pass
    return records

def store_simhash_csv(file_path, text_id, hash_int, hash_hex, text_length, text_hmac):
    with open(file_path, "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([text_id, hash_int, hash_hex, text_length, text_hmac, datetime.now().isoformat()])

# ---------- Main Function ----------
def add_text(text_id, text, silent=False):
    new_hash = simhash(text)
    text_hmac = compute_hmac(text)
    records = load_csv(CSV_FILE)
    
    closest_match = None
    max_similarity = -1
    for rec in records:
        sim, dist = simhash_similarity(new_hash, rec["simhash_int"])
        if sim > max_similarity:
            max_similarity = sim
            closest_match = {"id": rec["id"], "similarity": sim, "distance": dist}
    
    store_simhash_csv(CSV_FILE, text_id, new_hash, format(new_hash, '016x'), len(text), text_hmac)
    
    if not silent:
        if closest_match:
            print(f"{text_id}: sim={closest_match['similarity']:.3f} to '{closest_match['id']}' (dist={closest_match['distance']})")
        else:
            print(f"{text_id}: first entry")
    
    return {"id": text_id, "hash": new_hash, "hmac": text_hmac, "closest": closest_match}

# ---------- Batch Processing ----------
def process_batch(texts, verbose=False):
    results = []
    for text_id, text in texts:
        result = add_text(text_id, text, silent=not verbose)
        results.append(result)
    return results

# ---------- Example Usage ----------
def mainn():
    texts = [
        ("doc_01", "Hello world"),
        ("doc_02", "Hello brave new world"),
        ("doc_03", "Completely different text"),
        ("doc_04", "This is a random string combined together to form a sentence, in such a way that we are able to detect whether the text was generated from our end or not, which in turn further helps us to check whether it is AI Generated or Human Generated Text."),
        ("doc_05", "This is a just a string which is combined together to form a sentence, in such a way that we are able to detect whether the text had been generated from our end or not, which in turn further helps us to check whether it is AI Generated or Humanized/Generated Text.")
    ]
    
    print("Processing texts...")
    results = process_batch(texts, verbose=True)
    print(f"\n✅ Stored {len(results)} entries in {CSV_FILE}")

# maimn()

#writing SimHash & HMAC into the CSV File for Future Verification use
texts = [
    ("doc_01", Signature)
]
    
print("Processing texts...")
results = process_batch(texts, verbose=True)
print(f"\n✅ Stored {len(results)} entries in {CSV_FILE}")

print(f"\n\n\nOutput For you Prompt is: {to_out}\nNumber of char replaced: {to_out_numReplaced}")








