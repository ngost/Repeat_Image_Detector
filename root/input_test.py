ls = [1,2,3]

ls.append("1")
ls.append(set("hello"))
dic = {1:"key1",2 : "key2",3 : "key3"}
ls.append(dic)

print(type(ls.pop()))
print(type(ls.pop()))

by = bytes()