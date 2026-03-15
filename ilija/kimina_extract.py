import re
import json

def extract_proof_from_text(output: str) -> str:
    """
    Parse the proof from the model output for thinking models.
    Takes the last code inside ```lean4 and ``` that has the formal statement inside
    Args:
        output: The model output string.
    Returns:
        The parsed proof string.
    """
    lean4_codes = re.findall(r"```lean4\n(.*?)\n```", output, re.DOTALL)
    words = ["theorem", "by", ":=", "import"]

    for i in range(len(lean4_codes)):
        lean4_code = lean4_codes[-i - 1]
        if all(word in lean4_code for word in words):
            return lean4_code

    return ""

with open('kimina_good_raw.json', 'r') as f:
    data = json.load(f)
    for i, obj in enumerate(data):
        answers = []
        for ans in obj['answers']:
            extracted_proof = extract_proof_from_text(ans)
            if extracted_proof != "":
                stripped_mathlib = re.sub(r'^\s*import\s+Mathlib.*\n', '', extracted_proof, flags=re.MULTILINE)
                answers.append(stripped_mathlib)
        data[i]['answers'] = answers
    with open('kimina_good.json', 'w') as w:
        json.dump(data, w)