words = ['bat', 'rats', 'god', 'dog', 'cat', 'arts', 'star']
sort_words = {}
for word in words:
    sort_words[word] = ''.join(sorted(word))

print(sort_words)
anagrams = []
for i in range(len(words)):
    ana = [words[i]]
    for j in range(i + 1, len(words)):
        if sort_words[words[i]] == sort_words[words[j]]:
            ana.append(words[j])
    if len(ana) != 1:
        anagrams.append(ana)

print(anagrams)
