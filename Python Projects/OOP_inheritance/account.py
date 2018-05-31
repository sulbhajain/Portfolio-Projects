
# creating a class and defining functions with global and local variables.
# the script reads a account file for balance and updates the amount based on withdraw
# and deposit transactions.
# the code demostrates usage of default init() and del() methods for a class.

# class and methods declaration for Account:
class Account:

    def __init__(self, filepath):
        # here goes the initialization code
        self.filepath = filepath
        with open(filepath, 'r') as fh:
            self.balance = int(fh.read())

    def __del__(self):
        #called during exit
        print ("closing")

    def withdraw(self, amount):
        # withdraw the account
        self.balance = self.balance  - amount

    def deposit(self, amount):
        # add to the account
        self.balance = self.balance  + amount

    def commit(self):
        # update the balance to the file
        with open(self.filepath, "w") as fh:
            fh.write(str(self.balance))

class checking(Account):
    type = 'checking'

    def __init__(self, filepath, fee):
        Account.__init__(self,filepath)
        self.fee = fee

    def transfer(self,amount):
        self.balance = self.balance - amount - self.fee


# create Account object and access all methods to caryy out the operations

# create  Jack's account
jacks_checking = checking("jacks_checking.txt",1)
print ("Current Balance is ${}" .format(jacks_checking.balance))
# withdraw
jacks_checking.withdraw(100)
print ("Balance after withdrawing $100 is ${}".format(jacks_checking.balance))
# deposit
jacks_checking.deposit(200)
print ("Balance after depositing $200 is ${}".format(jacks_checking.balance))
# transfer
jacks_checking.transfer(300)
print ("Balance after transfer $300 is ${}".format(jacks_checking.balance))
# finally updating account balance in the file
jacks_checking.commit()
print ("Jacks checking type is {}".format(jacks_checking.type))


# create  John's account
johns_checking = checking("johns_checking.txt",1)
print ("Current Balance is ${}" .format(johns_checking.balance))
# withdraw
johns_checking.withdraw(10)
print ("Balance after withdrawing $10 is ${}".format(johns_checking.balance))
# deposit
johns_checking.deposit(20)
print ("Balance after depositing $20 is ${}".format(johns_checking.balance))
# transfer
johns_checking.transfer(30)
print ("Balance after transfer $30 is ${}".format(johns_checking.balance))
# finally updating account balance in the file
johns_checking.commit()
print ("Johns checking type is {}".format(johns_checking.type))
