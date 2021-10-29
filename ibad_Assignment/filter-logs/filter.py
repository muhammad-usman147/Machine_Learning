'''Note:
before execution, save your log file to a txt file
exp: file.log rename to ==> file.txt
'''

FILENAME = 'logs.txt' #log file name to read
SAVE_AS = 'updated_logs.txt' #new filtered file name, to store the new filtered data
with open(FILENAME,'r') as f:
  data = f.readlines()
new_data = ''
flag = True
order = 1
for i in data:
  if i != '\n':
    x = i.split("|")[4].strip()
    if x == '35=D':
      new_data += 'order: {} \n{}'.format(order,i)
      
      order += 1
    elif x == '35=8' or x == '35=F':
      new_data += "{} \n".format(i)

with open(SAVE_AS,'w') as f:
  f.write(new_data)
