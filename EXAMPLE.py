#!/usr/bin/env python3

from tfidf.corpus import Corpus

onion1 = '''Telling reporters they were having difficulty keeping track of all the new pastimes he was pursuing, friends of local man Mark Chapineau stated Tuesday that the recent divorcé was burning through hobbies at an unsustainable rate. “Last week, he was posting on Facebook about how he was getting into meditation, and now you can see he’s already onto black-and-white photography—boy, he’s really tearing through activities one after the other,” said Chapineau’s old college roommate Sahil Neela, adding that the 38-year-old insurance broker, whose five-year marriage ended in September, had, according to his latest status update, begun training for a marathon, despite mentioning a couple weeks prior that he had joined a rock-climbing gym. “It’s crazy. He’s throwing himself into cooking and collecting vinyl when he’s only had his microbrewing kit 24 hours. Seriously, if he doesn’t want to confront what just happened in his life, he’s going to have to space these things out a whole lot better.” Neela acknowledged, however, that he’d rather watch his friend exhaust every potential hobby than actually listen to him talk about his problems'''
onion2 = '''Directing the server to the large square in the corner, local 34-year-old Matthew Hinke asked for a big piece of cake during a workplace birthday party, sources confirmed Tuesday. “Can I get that big one right there? Yep, that one,” said the senior marketing manager, husband, and father of two while eagerly holding out his plastic plate in anticipation, having actively sought out the slice not only for its size but also because it had a full, intact icing flower on it. “Yeah, you got it. Perfect.” At press time, Hinke was making room on his plate for a big scoop of ice cream.'''
onion3 = '''Saying the small act of defiance helped to brighten her otherwise dejected mood these days, local woman Becca Curran told reporters Friday that stealing tampons from her office’s bathroom was currently her only source of joy. “Given the way everything’s been going lately, grabbing a handful of tampons and stuffing them into my bag has become the one thing I can really count on to lift my spirits,” said the 28-year-old billing specialist, who added that while the current sociopolitical climate makes it nearly impossible to feel optimistic about anything, purloining the feminine hygiene products every time she enters the office restroom remains a genuine pleasure and is reliably the highlight of her workday. “When I see that fully stocked basket sitting there on the countertop just ripe for the picking, it actually makes my day a bit better. I like knowing that I won’t have to pay for my own tampons and that I’m also taking advantage of my company’s resources. It’s really all I’ve got left to feel good about right now.” Curran added that she had no idea how the millions of women whose workplaces don’t provide free tampons were coping.'''

c = Corpus(gramsize=3, stopwords_file='tfidf/stopwords/english')
c['o1'] = onion1
c['o2'] = onion2
c['o3'] = onion3

# Extract the top 50 keywords from the 'o1' document
c.get_keywords(document_id='o1', tf_weight='binary', idf_weight='smooth', limit=50)

# Extract keywords from text you pass in, using the already made corpus
c.get_keywords(text='the large square in the corner', limit=5)
