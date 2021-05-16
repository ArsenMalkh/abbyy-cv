import streamlit
import os
from PIL import Image
from streamlit.script_runner import RerunException
from streamlit.script_request_queue import RerunData
import time
import SessionState
DATA_PATH = "../data/results"


def resize_to_fit(img):
    max_size = 500
    w, h = img.size
    mult = min(max_size / w, max_size / h)
    new_h = int(h * mult)
    new_w = int(w * mult)
    return img.resize((new_w, new_h))


def handle_image(path):
    image = Image.open(path)
    streamlit.image(resize_to_fit(image))


def find_files():
    return sorted([os.path.join(DATA_PATH, file) for file in os.listdir(DATA_PATH) if
                   file.split(".")[-1] in ['png', 'jpg', 'JPEG', 'JPG', 'jpeg']])


def main():
    streamlit.title('Detection checker tool')
    files = find_files()
    state = SessionState.get(GT=0, FOUND=0, TP=0, image_num=0, problems=[])

    if len(files) > state.image_num:
        file = files[state.image_num]
        streamlit.markdown(f"filename {file.split('/')[-1]}")
        handle_image(file)
        GT = int(streamlit.text_input("number of gt finder patterns", 0))
        FOUND = int(streamlit.text_input("number of found pattern (both correct and incorrect)", 0))
        TP = int(streamlit.text_input("number of correctly detected patterns", 0))
        streamlit.markdown(f"There are {len(files) - state.image_num} images to label")
        if streamlit.button("NEXT"):
            state.GT += GT
            state.FOUND += FOUND
            state.TP += TP
            state.image_num += 1
            if (FOUND != TP) or (TP != GT):
                state.problems.append(file.split('/')[-1])
            raise RerunException(RerunData())
    else:
        streamlit.markdown("There are no files left")
        try:
            streamlit.markdown(f'There are {state.image_num} images. Precision: {state.TP / state.FOUND:.2f} Recall: {state.TP / state.GT:.2f}')
            streamlit.markdown(f"problem files {str(state.problems)}")
        except:
            streamlit.markdown("There are no gt")



if __name__ == "__main__":
    main()
