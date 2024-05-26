class MyClass:
    
    def show(self,name):
        print("Hello", name)

    def input(self):
        user = input("User>> ")
        self.show(user)

# Create an instance of the class
obj = MyClass()
obj.input()
# Call function1, which in turn calls function2

