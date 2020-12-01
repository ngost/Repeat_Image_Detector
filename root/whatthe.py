# a 체크에서 b 체크로 이동
#1. a 체크에서 b 체크로 이동할때 x값 또는 y 값이 같은 경우, 직선상에 있기 때문에 부스터 한번으로 이동 가능
#2. 근처의 어떤 체크포인트로도 직선연결이 되어있지 않은경우,
#3. 걸어서 다른 체크포인트로 움직일 수 있는지?
#4. 못움직인다면, NO
#5. 움직일 수 있으면, 도착한 체크포인트에서 #1 반복
#2. 1번에 맞지 않은 경우, 근처 다른 체크포인트로 직선으로 갈 수 있는지?
#3.


class Checkpoint:
    x = 0
    y = 0

    def __init__(self, x, y):
        self.x = x
        self.y = y


def checkBoostSolution(start, end):
    if int(start.x) == int(end.x):
        return "YES"
    elif int(start.y) == int(end.y):
        return "YES"
    else:
        return "NO"


def calcQueryResult(start, end, life, checkpoint_list):
    if checkBoostSolution(checkpoint_list[start-1], checkpoint_list[end-1]) is "YES":
        return "YES"
    else:
        j = 0
        checkpoint_list2 = checkpoint_list
        checkpoint_list2.pop(start-1)
        checkpoint_list2.pop(end-1)
        while j < len(checkpoint_list2):
            j += 1
            result = checkBoostSolution(checkpoint_list[start-1], checkpoint_list2[j])
            if result is "YES":
                somthing = 0
        return




i = 0
checkpoint_list = []
checkpoint_tuple_set = set([])
checkpoint_num, query_num = input().split()

# init checkpoint
while i < int(checkpoint_num):
    i += 1
    input_x, input_y = input().split()
    checkpoint_tuple_set.add((input_x, input_y))

    if len(checkpoint_tuple_set) == i:
        checkpoint_list.append(Checkpoint(int(input_x), int(input_y)))
    else:
        i -= 1


# init query
i = 0
query_result = []
while i < int(query_num):
    start_check, end_check, life = input().split()
    query_result.append(calcQueryResult(int(start_check),int(end_check), life, checkpoint_list))
    i += 1
