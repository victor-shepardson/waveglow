import pandas as pd

data = pd.read_csv('../../data/mozilla_common_voice/eo/validated.tsv', sep='\t')

with open('common_voice_esperanto_validated.txt', 'w') as txt:
    for fname in data.path:
        txt.write('../../data/mozilla_common_voice/eo/clips/'+fname+'.mp3\n')
