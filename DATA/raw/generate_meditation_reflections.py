"""
Generate meditation_reflections.csv: 200+ non-duplicate rows, one column 'reflection_answer',
each row = 2-3 intra-coherent sentences on perceptual/pre-reflective aspects of meditation.
Uses lexicon typical of personal reports and phenomenological interviews.
Max 150 characters per sentence.
"""
import csv
import random
import os

# Meditation reflections: perceptual/pre-reflective aspects, phenomenological lexicon
# Each entry is 2-3 coherent sentences, each sentence <= 150 chars
REFLECTIONS = [
    # Bodily awareness & felt sense
    "I noticed a subtle heaviness in my chest that I had not felt before. It seemed to soften as I kept my attention there.",
    "The breath felt like it was happening on its own. I had the sense of being carried by it rather than controlling it.",
    "I became aware of a tingling in my fingertips. The sensation spread slowly up through my hands.",
    "There was a feeling of weight in my shoulders that I had not registered until I sat still. It seemed to shift as I breathed.",
    "I felt my body as a whole rather than in parts. The boundaries between inside and outside felt less sharp.",
    "A warmth arose in my belly and spread outward. I could not say whether it was physical or something else.",
    "My jaw had been clenched without my knowing. When I noticed it, the tension seemed to dissolve on its own.",
    "The contact of my feet with the floor became very vivid. I felt grounded in a way I had not expected.",
    "I noticed the weight of my hands resting on my thighs. The sensation changed as my attention moved to it.",
    "There was a sense of space opening up around my heart. I did not try to make it happen; it just arose.",
    "I felt the breath more in my back than in my chest. The movement was subtle but unmistakable.",
    "A slight tension in my forehead drew my attention. As I stayed with it, it seemed to soften and fade.",
    "The boundary between my body and the chair became unclear. I felt more like I was part of the room.",
    "I noticed a dull ache in my lower back that came and went. Observing it seemed to change its quality.",
    "My belly rose and fell without any effort from me. The rhythm felt steady and reassuring.",
    "There was a felt sense of something stuck in my throat. I stayed with it and it gradually eased.",
    "I became aware of the temperature of the air on my skin. It was cooler than I had realised.",
    "The soles of my feet felt alive with sensation. I had the impression of being rooted to the ground.",
    "A gentle pressure built behind my eyes. I did not resist it; I simply let it be there.",
    "I felt the weight of my head on my neck for the first time. It was both heavy and supported.",
    # Presence & temporality
    "Time seemed to slow down. Each moment felt longer and more spacious than usual.",
    "I had the sense of being fully present in a way I rarely am. The past and future did not pull at me.",
    "Each breath felt like a complete cycle in itself. I was not waiting for the next one.",
    "The moment seemed to stretch without any sense of urgency. I felt no need to move or do anything.",
    "I lost track of how long I had been sitting. The usual markers of time did not seem to apply.",
    "There was a quality of immediacy to everything. I felt directly in contact with what was happening.",
    "The usual stream of thoughts quieted. What remained was a simple sense of being here.",
    "I noticed a shift from doing to being. The distinction felt very clear in that moment.",
    "Each sound seemed to arrive fresh, without the overlay of memory. I heard things as if for the first time.",
    "The present moment felt thick with possibility. I had no sense of what would come next.",
    "I felt anchored in the here and now. The body was my reference point throughout.",
    "Time did not feel linear. Past and present seemed to overlap in a curious way.",
    "There was a stillness that was not the absence of movement. It felt more like a quality of presence.",
    "I experienced a gap between stimulus and response. In that gap, something else became possible.",
    "The usual sense of rushing forward dropped away. I felt content to be exactly where I was.",
    "Each inhalation and exhalation marked a small eternity. I was not counting; I was simply with them.",
    "I had the impression of stepping out of the stream of time. Everything slowed and clarified.",
    "The present felt vast rather than fleeting. I did not feel cramped or hurried.",
    "I noticed how often I was already in the next moment. Staying with this one required a gentle effort.",
    "There was a sense of timelessness without confusion. I knew where I was, but time did not press.",
    # Attention & noticing
    "My attention kept drifting, and I noticed that too. The noticing itself felt like a kind of return.",
    "I became aware of how quickly my mind jumped from one thing to another. There was no judgment, just observation.",
    "A thought arose and I saw it as a thought rather than as reality. That shift was subtle but significant.",
    "I noticed the difference between thinking about the breath and feeling it. The latter was more immediate.",
    "My awareness seemed to widen and narrow in waves. I did not try to control it.",
    "I caught myself planning while I was supposed to be resting. The catching felt like a moment of wakefulness.",
    "Attention felt like a beam that could be directed. When it wandered, I gently brought it back.",
    "I noticed how sensations changed when I paid attention to them. They were not fixed; they shifted.",
    "There was a sense of watching from a slight distance. I was both the one watching and the one being watched.",
    "I became aware of a background hum of mental activity. It did not stop, but it seemed less compelling.",
    "My attention rested more easily on the breath than usual. It did not feel forced.",
    "I noticed the moment of resistance before letting go. That resistance itself became an object of curiosity.",
    "Thoughts passed through like clouds. I did not need to hold on to them or push them away.",
    "I became aware of layers of experience. Sensation, thought, and feeling seemed to coexist without conflict.",
    "There was a quality of soft focus rather than sharp concentration. The edges of things were less defined.",
    "I noticed how my mind sought something to grasp. The seeking itself was interesting to observe.",
    "Attention felt like a kind of kindness. I was not drilling into experience; I was gently resting with it.",
    "I became aware of the space between thoughts. That space felt vast and quiet.",
    "I noticed the impulse to name or interpret what I felt. I let the impulse pass without following it.",
    "Attention seemed to have a texture. Sometimes it was sticky; sometimes it flowed freely.",
    # Spatial awareness & openness
    "I had the sense of space opening up inside my chest. It was not empty; it was spacious.",
    "The room around me seemed to expand. I felt less confined by the walls and ceiling.",
    "There was a quality of openness that I could not locate in the body. It felt more like a field of awareness.",
    "I noticed the space between sounds. The silence had a presence of its own.",
    "The boundaries of my body seemed to soften. I felt less like a solid object and more like a process.",
    "I had the impression of vastness behind my closed eyes. It was not dark; it was open.",
    "The space in front of me and behind me felt equal. I was in the middle of something larger.",
    "I felt a sense of expansion without movement. The body did not change, but the felt space did.",
    "There was a quality of airiness in my experience. Things felt less dense and heavy.",
    "The space between my breath and the next felt infinite. I rested in that gap.",
    "I became aware of the space that holds all experience. It seemed prior to the contents.",
    "The sense of being inside my head dropped away. Awareness felt more distributed.",
    "I noticed how the body occupies space without filling it completely. There was room around the edges.",
    "There was a felt sense of openness at the crown of my head. It was subtle but distinct.",
    "The space between thoughts felt like a landscape. I could rest there without needing anything.",
    "I had the impression of being held by space rather than floating in void. There was a quality of support.",
    "The boundaries between self and environment blurred. I was not sure where I ended and the room began.",
    "I felt less like a point and more like a region. Awareness seemed to have extension.",
    "The space inside and outside felt continuous. The usual division did not apply.",
    "There was a sense of spaciousness that did not depend on the size of the room. It felt internal and vast.",
    # Qualitative shifts & felt qualities
    "A sense of ease arose without my doing anything. It felt like a gift rather than an achievement.",
    "I noticed a quality of tenderness toward my own experience. It was not sentimental; it was steady.",
    "There was a feeling of rightness, as if things were as they should be. I did not need to change anything.",
    "A subtle joy emerged from somewhere I could not name. It was not excitement; it was quieter than that.",
    "I felt a kind of acceptance that was not resignation. It was more like a yes to what was present.",
    "There was a quality of clarity that was not intellectual. It felt more like a clearing of the senses.",
    "A sense of safety arose in my body. I had not realised I had been holding until it released.",
    "I noticed a feeling of fullness that was not heavy. It was more like enoughness.",
    "There was a gentleness in the way experience unfolded. Nothing felt harsh or forced.",
    "I felt a kind of trust in the process. I did not need to know what would happen next.",
    "A quality of freshness pervaded the experience. Things felt new rather than familiar.",
    "There was a sense of being met by the practice. I felt less alone in the sitting.",
    "I noticed a softening that was both physical and emotional. The two were hard to separate.",
    "There was a quality of allowance rather than effort. I was not trying; I was letting.",
    "A sense of wholeness arose. I did not feel fragmented the way I often do.",
    "I felt a kind of permission to be exactly as I was. No improvement was required.",
    "There was a quality of simplicity. Complexity dropped away without my pushing it.",
    "I noticed a feeling of being held. It was not by anyone or anything I could identify.",
    "A sense of peace arose without my seeking it. It was quiet and undramatic.",
    "There was a quality of lightness. The usual heaviness of the day had lifted.",
    # Embodiment & grounding
    "I felt more in my body than in my head. The shift was palpable.",
    "The ground beneath me felt solid and supportive. I could relax into it.",
    "I became aware of the vertical axis of my body. From the seat through the spine to the crown.",
    "There was a sense of being rooted through my sitting bones. The upper body felt free to rise.",
    "I felt the pull of gravity in a way that was comforting. It held me in place.",
    "The body felt like a reliable anchor. When the mind wandered, I could return to it.",
    "I noticed the alignment of my spine. Small adjustments seemed to change the whole experience.",
    "There was a sense of weight and substance in my torso. I felt substantial rather than scattered.",
    "I felt the body as a container for the breath. It expanded and contracted with each cycle.",
    "The connection between my body and the chair was vivid. I was aware of the support.",
    "I felt grounded through my legs and feet. The upper body could rest on that foundation.",
    "There was a sense of the body as a process rather than a thing. It was always changing.",
    "I noticed the difference between tension and ease in different regions. The map of the body became clearer.",
    "The body felt like a home I could return to. It was always there, waiting.",
    "I felt the flow of sensation through the body. Nothing was static.",
    "There was a sense of being fully embodied. The mind and body felt less separate.",
    "I noticed how the body responded to the breath without instruction. It knew what to do.",
    "The felt sense of the body was richer than I usually attend to. There was a lot going on.",
    "I felt anchored in the physical. That grounding allowed the mind to settle.",
    "There was a sense of the body as alive and responsive. It was not inert.",
    # Subtle sensations & micro-phenomena
    "I noticed a faint pulsation in my hands. I had never paid attention to it before.",
    "There was a subtle movement in my abdomen with each breath. It was not the chest; it was lower.",
    "I became aware of the temperature difference between in-breath and out-breath. The in-breath felt cooler.",
    "There was a faint ringing in my ears that I had been filtering out. When I noticed it, it seemed to shift.",
    "I felt a slight vibration or hum in my body. It was very subtle but consistent.",
    "I noticed the quality of the air as it entered my nostrils. It had a texture I had not registered.",
    "There was a sense of fluidity in my joints. They felt less fixed than usual.",
    "I became aware of the moment of transition between inhale and exhale. It was a kind of pause.",
    "There was a faint pressure at the tip of my tongue. I had no idea why it was there.",
    "I noticed the subtle movement of my eyes behind closed lids. They were not completely still.",
    "There was a sense of circulation or flow in my limbs. It was not dramatic; it was a suggestion.",
    "I became aware of the weight of my eyelids. They wanted to rest.",
    "There was a faint sense of expansion in my rib cage. It was more lateral than frontal.",
    "I noticed the moment when the breath turned from in to out. There was a quality of release.",
    "There was a subtle coolness at the back of my throat. I had not noticed it before.",
    "I felt a slight sway or micro-movement in my posture. The body was making small adjustments.",
    "There was a sense of the skin as a boundary. It was both porous and containing.",
    "I became aware of the rhythm of my heartbeat. It was slow and steady.",
    "There was a faint tingling at the crown of my head. It came and went.",
    "I noticed the quality of contact between my hands. Were they touching? The sensation was ambiguous.",
    "There was a subtle sense of alignment that came and went. The body was finding its balance.",
    # Sound & environmental awareness
    "I became aware of sounds without identifying them. They were just present.",
    "The silence between sounds felt as substantial as the sounds themselves. I rested in both.",
    "I noticed how my mind tried to name and locate each sound. I let that habit relax.",
    "There was a quality of openness to whatever arose. No sound was an interruption.",
    "I felt the sounds in my body as well as in my ears. Some resonated in my chest.",
    "The distinction between inner and outer sound softened. I was not sure where some sounds originated.",
    "I became aware of the layers of sound. Near and far, loud and soft, all at once.",
    "There was a sense of sound as texture. Each one had a different quality.",
    "I noticed how sounds came and went without my holding on. Letting them pass felt natural.",
    "The ambient noise of the room became part of the practice. I did not need silence.",
    "I felt a shift from hearing to listening. The latter felt more receptive.",
    "There was a quality of acceptance of the soundscape. I was part of it, not separate.",
    "I became aware of the space that sound travels through. It seemed vast.",
    "There was a sense of being immersed in the auditory field. I was not outside it, observing.",
    "I noticed how some sounds triggered thoughts and others did not. The difference was interesting.",
    "The sound of my own breath became part of the landscape. It was one among many.",
    "There was a quality of transparency to the sounds. I could hear through them to the silence.",
    "I became aware of sounds I had been filtering out. They had been there all along.",
    "There was a sense of sound as event rather than object. Each one was momentary.",
    "I noticed the way sound faded and left a kind of echo. The absence was also present.",
    # Breath & respiratory awareness
    "The breath found its own rhythm without my guiding it. I was surprised by how natural it felt.",
    "I noticed the breath in different parts of my body. Sometimes the belly, sometimes the chest, sometimes the nose.",
    "There was a quality of the breath as teacher. It showed me when I was striving.",
    "I became aware of the natural pause at the end of the exhale. I did not need to rush to the next inhale.",
    "The breath felt like a bridge between body and mind. I could use it to return when I drifted.",
    "I noticed how the breath changed when I paid attention. It often became slower and softer.",
    "There was a sense of the breath as movement. The whole body participated in a subtle way.",
    "I became aware of the texture of the breath. Sometimes smooth, sometimes jagged.",
    "The breath seemed to breathe itself. I was the witness rather than the doer.",
    "I noticed the moment of allowing before each inhale. The body knew when to breathe.",
    "There was a quality of intimacy with the breath. It was close and immediate.",
    "I became aware of how little I usually notice my breath. It had been happening in the background all along.",
    "The breath felt like an anchor. When thoughts pulled me away, I could return to it.",
    "I noticed the difference between shallow and deep breathing. The body preferred a middle ground.",
    "There was a sense of the breath as life itself. Simple, constant, reliable.",
    "I became aware of the breath in the context of the whole body. It was not isolated.",
    "The exhale felt like a release of something I had been holding. I did not know what.",
    "I noticed how the breath responded to my mental state. When I relaxed, it deepened.",
    "There was a quality of the breath as gift. I had not earned it; it was given.",
    "I became aware of the breath as rhythm. It had a music of its own.",
    "The breath felt like a companion. I was not alone in the sitting.",
    # Self & identity
    "I had the sense of being both the observer and the observed. The distinction blurred.",
    "The usual sense of 'me' as a solid center felt less fixed. I was more like a process.",
    "I noticed how often I was identified with my thoughts. When I saw that, something loosened.",
    "There was a quality of witnessing that did not feel like a separate self. It was more like awareness itself.",
    "I felt less like a noun and more like a verb. I was happening rather than being a thing.",
    "The boundary between self and experience seemed porous. I was in the experience, not behind it.",
    "I noticed the construction of the sense of 'I'. It arose in relation to thoughts and sensations.",
    "There was a feeling of spaciousness where the sense of self usually sits. It was not empty; it was open.",
    "I had the impression of being larger than my usual identity. The edges had softened.",
    "The sense of being a separate self felt like a habit. In moments, it dropped away.",
    "I noticed how the 'I' was different in different contexts. It was not one fixed thing.",
    "There was a quality of transparency to the sense of self. I could see through it.",
    "I felt less identified with my body. It was more like something I was inhabiting.",
    "The usual narrative of who I am seemed to pause. What remained was simpler.",
    "I noticed the effort of maintaining a coherent self-story. In stillness, that effort relaxed.",
    "There was a sense of being prior to the personal. I could not quite name it.",
    "I felt the constructed nature of identity. It was made of moments, not substance.",
    "The sense of 'me' and 'not me' seemed to arise together. They were not fundamentally separate.",
    "I noticed how the self reasserted itself after moments of openness. The return was gradual.",
    "There was a quality of mystery at the heart of experience. I did not need to resolve it.",
]


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences (simple period-based split)."""
    return [s.strip() for s in text.split(". ") if s.strip()]


def _all_sentences_under_limit(entry: str, max_chars: int = 150) -> bool:
    """Check that each sentence is at most max_chars."""
    sentences = _split_into_sentences(entry)
    return all(len(s) <= max_chars for s in sentences)


def main():
    random.seed(42)
    # Filter to entries with 2-3 sentences, each <= 150 chars
    valid = [r for r in REFLECTIONS if 2 <= len(_split_into_sentences(r)) <= 3 and _all_sentences_under_limit(r)]
    # Sample without replacement to ensure no duplicates
    k = min(210, len(valid))
    rows = random.sample(valid, k)
    random.shuffle(rows)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meditation_reflections.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["reflection_answer"])
        for r in rows:
            w.writerow([r])
    print(f"Wrote {len(rows)} non-duplicate rows to {out_path}")


if __name__ == "__main__":
    main()
