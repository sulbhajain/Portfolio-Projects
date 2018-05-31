
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

# create Account object and access all methods to caryy out the operations

# create account
account = Account("balance.txt")
print ("Current Balance is ${}" .format(account.balance))

# withdraw
account.withdraw(100)
print ("Balance after withdrawing $100 is ${}".format(account.balance))

# deposit
account.deposit(200)
print ("Balance after depositing $200 is ${}".format(account.balance))

# finally updating account balance in the file
account.commit()
