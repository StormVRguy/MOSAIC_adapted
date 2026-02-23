"""
Generate trial_200_topics.csv: 200+ rows, one column 'reflection_answer',
each row = 2+ related sentences, randomly over 5 topics.
"""
import csv
import random

# 5 topics, each with multiple 2-sentence pairs (related and coherent)
TOPICS = [
    # AI computation
    [
        "Training the model took three days on four GPUs. We used mixed precision to save memory.",
        "The inference pipeline runs in under fifty milliseconds. Batch size was tuned for our hardware.",
        "Neural network layers were initialized with He initialization. Gradient clipping prevented exploding gradients.",
        "The dataset was split into train and validation sets. We applied standard normalization to the features.",
        "Hyperparameter search ran for two weeks on a cluster. Learning rate and batch size had the biggest impact.",
        "We used transfer learning from a pretrained encoder. Fine-tuning took only a few hours.",
        "The loss curve flattened after epoch twenty. Early stopping would have saved compute.",
        "Attention weights were visualized for interpretability. Some heads clearly focused on key tokens.",
        "Quantization reduced the model size by seventy-five percent. Accuracy dropped by less than one percent.",
        "Distributed training scaled to thirty-two nodes. Communication overhead was the main bottleneck.",
        "We logged metrics to TensorBoard every hundred steps. The learning rate schedule was cosine decay.",
        "Data augmentation doubled the effective dataset size. We used random cropping and horizontal flips.",
        "The transformer has twelve layers and twelve attention heads. Embedding dimension is seven hundred sixty-eight.",
        "We benchmarked inference on CPU and GPU. The GPU was forty times faster.",
        "Overfitting started around epoch fifteen. We added dropout and weight decay.",
        "The API serves predictions via a REST endpoint. We added rate limiting for production.",
        "Feature importance was computed with SHAP values. The top three features explained most of the variance.",
        "We switched from Adam to AdamW for better generalization. Weight decay is now decoupled from the gradient.",
        "The model was exported to ONNX for deployment. Inference is now framework-agnostic.",
        "Cross-validation gave a more reliable estimate of performance. We used five folds with stratification.",
    ],
    # Recipes for biscuits
    [
        "Cream the butter and sugar until light and fluffy. Then beat in the egg and vanilla.",
        "Sift the flour and baking powder into a bowl. Add a pinch of salt and mix well.",
        "Chill the dough for at least an hour before rolling. This prevents the biscuits from spreading too much.",
        "Cut the biscuits with a round cutter and place on a lined tray. Leave a little space between each one.",
        "Bake for twelve minutes until the edges are golden. Let them cool on the tray for five minutes.",
        "The recipe calls for cold butter cut into small cubes. Rubbing it into the flour gives a crumbly texture.",
        "You can add chocolate chips or dried fruit to the mix. Fold them in gently at the end.",
        "Brush the tops with a little milk before baking. This gives a nice shine when they come out.",
        "Store the biscuits in an airtight tin for up to a week. They also freeze well for a month.",
        "Use brown sugar if you want a chewier biscuit. White sugar makes them crisper.",
        "The oven should be preheated to one hundred eighty degrees. A hot oven sets the shape quickly.",
        "Roll the dough to about half a centimetre thick. Thicker biscuits need a few more minutes in the oven.",
        "Add a teaspoon of cinnamon for a warm flavour. Nutmeg also works well with shortbread.",
        "Replace some of the flour with ground almonds for a richer taste. The texture will be slightly denser.",
        "Dip the cooled biscuits in melted chocolate if you like. Leave them to set on baking paper.",
        "Buttermilk makes the biscuits tender and slightly tangy. You can use milk and lemon juice as a substitute.",
        "Press the dough into a slab and score before baking. Break into pieces when cool.",
        "Golden syrup gives a soft and chewy result. Black treacle will make them darker and stronger.",
        "Add oats for a heartier biscuit that keeps you full. They go well with a cup of tea.",
        "The key is not to overwork the dough. Mix until just combined for a tender crumb.",
    ],
    # Birth family celebration
    [
        "The baby shower was held in the garden on a sunny Saturday. Everyone brought a small gift for the newborn.",
        "My sister gave birth to a healthy boy last week. The whole family visited the hospital the next day.",
        "We decorated the hall with balloons and a welcome banner. The cake had the baby name and birth date on it.",
        "Grandma knitted a blanket and a set of booties. She said it took her two months to finish.",
        "The guests wrote wishes for the baby in a keepsake book. We will give it to him when he is older.",
        "We ordered catering for about forty people. There was plenty of food and drink for everyone.",
        "My brother could not stop smiling when he held his daughter. It was his first time as a father.",
        "We took lots of photos of the new family of three. Those pictures are already on the wall at home.",
        "The baby received so many clothes and toys. We had to sort them by age and size.",
        "A close friend offered to help with the first few nights. New parents really need that support.",
        "The maternity ward was full but the staff were wonderful. We left a thank-you card and some chocolates.",
        "We planned a naming ceremony for the following month. Relatives flew in from abroad for the occasion.",
        "The baby looked like his father from the first day. Same nose and same calm expression.",
        "We set up a corner with nappies and wipes for the guests. Several people offered to change the baby.",
        "The speech from the new dad brought everyone to tears. He thanked his wife and both families.",
        "We collected donations for a children charity instead of more gifts. The parents had everything they needed.",
        "The midwife came for a home visit a week later. She checked the baby and gave feeding advice.",
        "We framed the hand and foot prints from the hospital. They sit on the shelf in the living room.",
        "The older cousins were so excited to meet the baby. They took turns holding him with supervision.",
        "We kept the celebration simple so the mother could rest. She had had a long labour.",
    ],
    # Murano glass-making
    [
        "Murano glass has been made in Venice for over seven hundred years. The island was once the heart of European glass production.",
        "The master heats the glass in a furnace at over a thousand degrees. He then shapes it with tools and breath.",
        "Each piece is hand-blown and therefore unique. No two vases or figures are exactly alike.",
        "The craftsmen use soda-lime glass and add minerals for colour. Cobalt gives blue and gold leaf adds shine.",
        "Lampworking is another technique used for smaller items. The artisan works at a torch with rods of glass.",
        "The canes are made by layering coloured glass and pulling them thin. Slices reveal a pattern when cut crosswise.",
        "Murano beads were traded across the Mediterranean for centuries. They have been found in archaeological sites far from Italy.",
        "Apprentices train for many years before they can work alone. The skills are often passed down in families.",
        "The furnace must stay lit day and night during the season. Cooling it down would damage the structure.",
        "Chandeliers from Murano hang in palaces and hotels worldwide. They can have hundreds of glass elements.",
        "The glass is sometimes rolled in gold or silver leaf while hot. This creates a luxurious finish.",
        "Modern Murano artists still use traditional methods but also experiment. Some combine glass with other materials.",
        "The island has a museum dedicated to glass history. Visitors can watch live demonstrations there.",
        "Export of glassmaking knowledge was once forbidden by the Republic. Craftsmen who left could face severe punishment.",
        "Aventurine glass contains copper crystals that sparkle when polished. It was discovered by chance in Murano.",
        "The maestri often work in teams with assistants handling the blowpipe. Timing and coordination are essential.",
        "Glass from Murano is still signed or labelled by many workshops. Collectors look for these marks.",
        "The cooling process must be slow to avoid cracking. Pieces rest in an annealing oven for hours.",
        "Millefiori means a thousand flowers and refers to the patterned canes. They are fused together to form the design.",
        "Tourists can take short courses to try glassmaking themselves. It is much harder than it looks.",
    ],
    # University exams
    [
        "The final exam covered everything from week one to week twelve. I spent the last three days revising the notes.",
        "I sat the exam in a large hall with hundreds of other students. Invigilators walked between the rows the whole time.",
        "The first question was on theory and the second was a problem set. I ran out of time on the last part.",
        "Results are released online two weeks after the exam period. I refreshed the page every hour on the day.",
        "I failed the midterm but passed the resit in the summer. The extra study time really helped.",
        "The lecturer gave a revision session the week before the exam. She went through the past papers and answered questions.",
        "We are not allowed to bring any notes or devices into the room. Only a clear pencil case and water bottle.",
        "The exam was two hours long with ten questions. I left two questions partly unanswered.",
        "I formed a study group with three friends from the course. We met in the library every evening.",
        "The marking scheme was published after the results came out. I lost marks for not showing my working.",
        "I had to sit four exams in five days. The last one was the hardest because I was tired.",
        "Plagiarism in exams leads to serious disciplinary action. The university uses software to check submissions.",
        "I requested extra time because of a documented learning difference. The disability office arranged it quickly.",
        "The exam questions were similar to the tutorial exercises. I was glad I had done all of them.",
        "I could not read my own handwriting in one answer. I hope the marker could decipher it.",
        "The module had a forty percent coursework and sixty percent exam split. I did better on the coursework.",
        "We get one resit attempt per module if we fail. After that we have to repeat the whole module.",
        "The exam board meets in July to confirm all the grades. They can adjust boundaries in special cases.",
        "I revised using flashcards and past papers. Active recall worked better than just rereading.",
        "The invigilator announced ten minutes left and I was still on question seven. I had to rush the rest.",
    ],
]

def main():
    random.seed(42)
    rows = []
    for _ in range(210):  # 210 rows to be sure we have 200+
        topic = random.choice(TOPICS)
        row_text = random.choice(topic)
        rows.append(row_text)
    random.shuffle(rows)  # random order of topics
    import os
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trial_200_topics.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["reflection_answer"])
        for r in rows:
            w.writerow([r])
    print(f"Wrote {len(rows)} rows to {out_path}")

if __name__ == "__main__":
    main()
