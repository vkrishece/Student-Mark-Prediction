from __future__ import division
import csv
import random
from random import shuffle

n=10000

sno=['Student ID']
for j in range(n):
    sno.append(j+1)

reg=[]
for j in range(9000):
     reg.append('Regular')

lat=[]
for j in range(1000):
     lat.append('Lateral')



admi=['Regular/Lateral']
adm=reg+lat
adm1 = random.sample(adm, len(adm)) # Copy and shuffle
asmi=admi+adm1
#print asmi



a=['CIA-1']

for j in range(n):
    a.append(random.randint(1,50))
#print('Randomised list is: ',a)

b=['CIA-2']
for j in range(n):
    b.append(random.randint(1,50))
#print('Randomised list is: ',b)

c=['CIA-3']
for j in range(n):
    c.append(random.randint(1,50))

assign1=['Assignment-1']
for j in range(n):
    assign1.append(random.randint(8,10))

assign2=['Assignment-2']
for j in range(n):
    assign2.append(random.randint(8,10))

assign3=['Assignment-3']
for j in range(n):
    assign3.append(random.randint(8,10))


attendance=['Attendance (%)']
for j in range(n):
    attendance.append(random.randint(75,100))

internal=['Internal(50)']
for j in range(n):
    m=a[j+1]/2.5 , b[j+1]/2.5, c[j+1]/2.5
    testMarks=list(m)
    y=testMarks.index(min(testMarks))
    maxMarks=sum(testMarks[:y]+testMarks[y+1:])
    #print a[j+1],b[j+1],c[j+1],m,maxMarks
    jj=assign1[j+1]+assign2[j+1]+assign3[j+1]
    byCIA=int(maxMarks+(jj/3))
    internal.append(byCIA)

external=['External(100)']
for j in range(n):
    external.append(random.randint(40,100))

total=['Final Marks']
for j in range(n):
    totalMarks=int(internal[j+1]+(external[j+1]/2))
    total.append(totalMarks)

grade=['Grade']
for j in range(n):
    if(total[j+1] > 90):
	val='S'
    elif(total[j+1] > 85):
	val='A+'
    elif(total[j+1] > 80 ):
	val='A'
    elif(total[j+1] > 70):
	val='B'
    elif(total[j+1] > 60 ):
	val='C'
    elif(total[j+1] >= 50):
	val='D'
    elif(total[j+1] < 50):
	val='F'
    else:
	val='E'
    #print total[j+1],val
    grade.append(val)

rows = zip(sno,asmi,a,b,c,assign1,assign2,assign3,attendance,internal,external,total,grade)




with open('dataset.csv', 'wb') as f:
    writer = csv.writer(f)
    for row in rows:
    	writer.writerow(row)

"""
with open('test.csv', 'r') as f:
     for line in f:
         print line
"""

