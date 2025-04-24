def to_gloss(text, names=[]):
    # Step 1: Normalize and split the text
    words = text.strip().split()
    glossed = []

    for word in words:
        clean_word = word.strip(".,?!").lower()
        if clean_word in [n.lower() for n in names]:
            # Fingerspell the name
            fingerspelled = ' '.join(list(clean_word.upper()))
            glossed.append(fingerspelled)
        else:
            glossed.append(clean_word.upper())

    return ' '.join(glossed)

# Example usage:
sentence = "Yesterday Wyatt went to the store."
names = ["Wyatt"]
gloss = to_gloss(sentence, names)
print(gloss)
