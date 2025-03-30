# tot_search.py
from .tot_evaluation import evaluate_emotion_reasoning

EMOTION_LABELS = ['Neutral', 'Happy', 'Sad', 'Angry', 'Frustrated', 'Excited']

def bfs_emotion_search(top_text, top_audio, top_vision):
    best_emotion = None
    best_score = -1
    best_reasoning = ""

    # BFS over combinations (simple exhaustive search)
    for t_idx in top_text:
        for a_idx in top_audio:
            for v_idx in top_vision:
                text_emotion = EMOTION_LABELS[t_idx]
                audio_emotion = EMOTION_LABELS[a_idx]
                visual_emotion = EMOTION_LABELS[v_idx]

                reasoning = evaluate_emotion_reasoning(text_emotion, audio_emotion, visual_emotion)
                print(f"Evaluated {text_emotion}, {audio_emotion}, {visual_emotion}: {reasoning}")

                score = heuristic_score(reasoning)
                if score > best_score:
                    best_score = score
                    best_emotion = (text_emotion, audio_emotion, visual_emotion)
                    best_reasoning = reasoning

    return best_emotion, best_reasoning

def heuristic_score(reasoning_text):
    keywords = ['most accurate', 'consistent', 'dominant', 'clearly']
    return sum(kw in reasoning_text.lower() for kw in keywords)
