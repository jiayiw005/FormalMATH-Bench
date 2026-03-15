import re
import json

verified_thms = 0
total_errs = 0

linarith_misuse = 0
rewrite = 0
no_goals = 0
type_mismatch = 0
unknown = 0
unexpected_token = 0
omega_misuse = 0
#linarith simplex algo timeout
timeout = 0
other_tactic_fail = 0
sorries = 0
unsolved_goals = 0

with open('messages.txt', 'w') as out:
    with open('output/results.jsonl', 'r') as f:
        for line in f:
            obj = json.loads(line)
            verified = False
            for answer in obj["answers"]:
                if answer['verified']:
                    verified = True
                for msg in answer["messages"]:
                    if msg['severity'] == 'error':
                        text = msg['data']
                        if 'linarith' in text:
                            linarith_misuse += 1
                        elif 'rewrite' in text:
                            rewrite += 1
                        elif 'Type mismatch' in text:
                            type_mismatch += 1
                        elif 'Unknown' in text:
                            unknown += 1
                        elif "unexpected" in text:
                            unexpected_token += 1
                        elif "omega" in text:
                            omega_misuse += 1
                        elif "heartbeat" in text:
                            timeout += 1
                        elif "` failed" in text:
                            other_tactic_fail += 1
                        elif 'No goals to' in text:
                            no_goals += 1
                        elif 'sorry' in text:
                            sorries += 1
                        elif 'unsolved goals' in text:
                            unsolved_goals += 1
                        out.write(msg['data'] + '\n')
                        total_errs += 1
            if verified:
                verified_thms += 1
print("Verified theorems", verified_thms)
print("Total errors", total_errs)
print("Linarith misuse", linarith_misuse)
print("Omega misuse", omega_misuse)
print("Timeout", timeout)
print("No goals to be solved", no_goals)
print("Rewrite misuse", rewrite)
print("Type mismatch", type_mismatch)
print("Unknown identifier/constant", unknown)
print("Unexpected token", unexpected_token)
print("Other tactic failures", other_tactic_fail)
print("Sorries", sorries)
print("Unsolved goals", unsolved_goals)
# 47% kimina