file_data=""
s="../co3d-main/dataset/"
with open("text.txt",'r') as f:
    for line in f:
        if(s in line):
            line=line.replace(s,'')
        file_data+=line
with open("text.txt",'w') as f:
    f.write(file_data)
