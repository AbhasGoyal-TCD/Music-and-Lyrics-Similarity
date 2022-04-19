import glob
import os


working_dir = 'D:\\TCD\\Text Analytics\\Group Assignment\\Music-and-Lyrics-Similarity\\Database\\'

os.chdir(working_dir)

midi_files = glob.glob('*\\MIDI\\*\\*.mid')

for file in midi_files:
    data = []
    split_values = file.split('\\')
    
    if not os.path.exists(split_values[0] +  '\\ABC\\' +  split_values[2]):
        os.makedirs(split_values[0] +  '\\ABC\\' +  split_values[2])
    
    #command = 'midi2abc "' + working_dir  + file + '"' ' -o "' + working_dir + split_values[0] +  '\\ABC\\' +  split_values[2] + '\\' + split_values[3].replace('mid','abc') + '"'
    #print(command)
    
    #os.system("cmd /c 'D:\\TCD\\Text Analytics\\Group Assignment\\Music-and-Lyrics-Similarity\\midiabc\\midi2abc") # + '" -o "' + working_dir + '\\' + split_values[0] +  '\\ABC\\' +  split_values[2] + '\\' + split_values[3].replace('mid','abc'))

