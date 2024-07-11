a=(1,2,3)
a

a=[1,2,3]
a

# soft copy
b= a
a

a[1]=4
a

b
id(a)
id(b)

# deep copy
a=[1,2,3]
a

b=a[:] # method 1
b=a.copy() # method 2

a[1]=4
a
b
