import os
import json
from blip.demo import blip_run
path="./co3d"
os.environ['NO_PROXY'] = 'huggingface.co' 
g=os.listdir(path)
f=open("full_text.txt",'w')
for dir in g:
    if(dir[0]!='_' and '.' not in dir):
        seq_path=path+'/'+dir+'/set_lists'
        seqs=os.listdir(seq_path)
        for seq in seqs:
            with open(seq_path+'/'+seq,'r') as file:
                seq_info=json.load(file)
                train_data=seq_info["train"]
                old_path=''
                for data in train_data:
                    new_path=data[0]
                    if(new_path!=old_path):
                        text=blip_run(path+'/'+data[2])
                        f.write(dir+'/'+new_path+"/images")
                        f.write('\n')
                        f.write(text)
                        f.write('\n')
                        print(dir+'/'+new_path+"/images")
                        print(text)
                        old_path=new_path
f.close()


                    

