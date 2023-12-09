import json
import os
import sys

all_rare_words = set()
dirname = sys.argv[1]

def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

def find_rareword(sent, rarewords):
    blist = []
    for word in rarewords:
        if isEnglish(word):
            if  (f' {word} ' in sent) or \
                (f'{word} ' == sent[:len(word) + 1]) or \
                (f' {word}' == sent[-(len(word) + 1):]) or \
                (word == sent):
                blist.append(word)
        else:
            if word in sent.replace(' ', ''):
                blist.append(word)
    return blist

with open("local/esun.entity.txt") as fin:
    for line in fin:
        all_rare_words.add(line.strip())

perutt_blist = {}
with open(os.path.join(dirname, "text")) as fin:
    for line in fin:
        uttname = line.split()[0]
        content = ' '.join(line.split()[1:])
        print(content)
        perutt_blist[uttname] = find_rareword(content, all_rare_words)
        # for bword in all_rare_words:
        #     if bword in content and bword not in perutt_blist[uttname]:
        #         perutt_blist[uttname].append(bword)

with open(os.path.join(dirname, "perutt_blist.json"), "w") as fout:
    json.dump(perutt_blist, fout, indent=4, ensure_ascii=False)
