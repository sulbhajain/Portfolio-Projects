# Dictionry lookup progrm which accepts a string and returns the meaning/s. The program
# uses json database for word meaning reference.

import json

from difflib import get_close_matches

file = json.load(open("data files/data.json"))

def translate(word):

    wrd = word.lower()
    if wrd in file:
        return (file[wrd])

    elif wrd.title() in file: # To check if words e.g. Texas exist instead of 'texas' in the Dictionry
        return (file[wrd.title()])

    elif wrd.upper() in file: # To check if words e.g. USA exist insteadof 'usa' in the Dictionry
        return (file[wrd.upper()])

    elif len(get_close_matches(wrd, file.keys())) > 0: # to check if any closly matching word is there in the dict
        close_match = get_close_matches(wrd, file.keys())[0] # select first(0) closest match to the input word
        print("the words matches closly with %s" % close_match)
        input_res = input("Press Y for yes, N for no:")
        input_res = input_res.lower()
        if input_res == 'y':

            return translate(close_match)
        else:
            return("No such word in the dic")
    else:
        return("Word does not exist")

word = input("Enter the word for meaning ")
meaning_list = translate(word)
if type(meaning_list) ==list:
    for item in meaning_list:
        print (item)
else:
    print(meaning_list)
