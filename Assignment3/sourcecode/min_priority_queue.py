#coding:utf-8

class PriorityNode:

    def __init__(self, key, val):
        '''
            init function of Priority Node

            @key: value of priority
            @val: value of node
        '''
        self.key = key
        self.val = val

class MinPriorityQueue:

    def __init__(self, max_length, comparer):
        '''
            init function of MinPriorityQueue

            @max_length: int; maximun length of queue
            @comparer: callable; used to compare two nodes; Node A and B, A>B => return>0; A==B => return=0; A <= B => return < 0;
        '''
        self.max_length = max_length
        self.node_size = 0
        self.comparer = comparer
        self.queue = [ None for i in xrange(max_length) + 1]
        self.is_heap = False

    def insert(self, node):
        '''
            add a node to the tail of the queue but not to maintain the heap

            @node: Priority Node
        '''
        if self.is_heap:
            raise Exception('Already a priority, please use push')

        if self.node_size == self.max_length:
            raise Exception('Overflow')
        self.node_size += 1
        self.queue[self.node_size] = node
        
    def heapify(self, i):
        '''
            heapify a node from top to down

            @i: index of node in self.queue
        '''
        if not self.is_heap:
            raise Exception('Not a priority queue yet')
        
        if i <= 0:
            raise Exception('Heapify a invalid index, index=%d, node_size=%d, max_length=%d'%(i, self.node_size, self.max_length))
        
        current_i = i
        while True:
            left_chd = current_i * 2
            right_chd = left_chd + 1
            smaller = current_i
            
            if left_chd<self.node_size and self.comparer(self.queue[left_chd], self.queue[smaller]) < 0:
                smaller = left_chd
            if right_chd<self.node_size and self.comparer(self.queue[right_chd], self.queue[smaller]) < 0:
                smaller = right_chd
            
            if smaller != current_i:
                #swap
                tmp = self.queue[smaller]
                self.queue[smaller] = self.queue[current_i]
                self.queue[current_i] = tmp
                current_i = smaller
            else:
                break

    def build_queue(self):
        '''
            build the queue into a min priority queue (heap)
        '''
        tail = self.node_size / 2
        for i in xrange(tail, 0, -1):
            self.heapify(i)
        self.is_heap = True

    def pop(self):
        '''
            return the head of the priority and remove it

            #return: PriorityNode
        '''
        if not self.is_heap:
            raise Exception('Not a priority queue yet')

        if self.node_size == 0:
            raise Exception('queue empty')
        rlt = self.queue[1]
        #swap
        self.queue[1] = self.queue[self.node_size]
        self.queue[self.node_size] = None
        self.node_size -= 1
        #build
        self.build_queue()
        
        return rlt

    def push(self, node):
        '''
            push a node to priority

            @node: Priority Node
        '''
        if not self.is_heap:
            raise Exception('Not a priority queue yet')

        if self.node_size == self.max_length:
            raise Exception('priority full')
        
        self.node_size += 1
        self.queue[self.node_size] = node
        current_i = self.node_size
        while True:
            parent = current_i / 2
            if parent == 0:
                break
            if self.comparer(self.queue[parent], self.queue[current_i]) > 0:
                tmp = self.queue[parent]
                self.queue[parent] = self.queue[current_i]
                self.queue[current_i] = tmp
                current_i = parent
            else:
                break

    def remove_all(self, idx_list):
        '''
            remove all node in idx_list

            @idx_list: list, the index list of nodes to remove
        '''
        if not self.is_heap:
            raise Exception('Not a priority queue yet')
        
        #remove
        for idx in idx_list:
            self.queue[idx] = -1
        self.queue.remove(-1)
        self.node_size = self.node_size - len(idx_list)
        self.queue.extend([None for i in xrange(len(idx_list))])
        #build heap again
        self.build_queue()
