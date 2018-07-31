import spacy,os
from textacy.keyterms import sgrank,textrank


"""
Any necessary pre-processing for the keywords file. For now, just stripping newlines, tabs and lowercasing.
"""
def process(str):
    str = str.strip().replace("\n","").replace("\t","").lower()
    return str

"""
Purpose: Extract texts and manually assigned keyphrases from the INSPEC dataset.
Output: Tuple of 2 items. 
First item is a list containing text content of each file.
Second item is a list containing a list of keyphrases for each text.
Dataset obtained from: https://github.com/snkim/AutomaticKeyphraseExtraction
(https://github.com/snkim/AutomaticKeyphraseExtraction/blob/master/Hulth2003.tar.gz)
"""
def get_inspec():
    inspec_folder = "Hulth2003.tar\\Test"
    inspec_files = os.listdir(inspec_folder)
    result = []
    for myfile in inspec_files:
        if myfile.endswith("abstr"): #This is the actual content.
            content = open(os.path.join(inspec_folder,myfile),encoding="utf-8",errors="ignore").read().strip()
            keywords_file = myfile.replace(".abstr",".uncontr")
            keywords =  [process(item) for item in open(os.path.join(inspec_folder,keywords_file),encoding="utf-8",errors="ignore").read().strip().split(";")]
            result.append((content,keywords))
    return result

inspec = get_inspec()
print(len(inspec))
spacymodel = spacy.load('en_core_web_sm', disable=('parser','ner'))
fw = open("tempKPEout-allinspec-textrank.txt","w")
for content,keywords in inspec:
        fw.write(content+"\n")
        fw.write(str(keywords))
        fw.write("\n")
        spacified = spacymodel(content)
        fw.write(str([term[0] for term in textrank(spacified)]))
        fw.write("\n\n")
fw.close()
