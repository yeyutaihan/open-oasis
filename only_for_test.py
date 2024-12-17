class A:
    def __init__(self):
        print("A")
        self.a = 1
        
    @property
    def aa(self):
        return self.a

if __name__ == "__main__":
    Aa = A()
    print(Aa.aa)