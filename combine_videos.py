import os
import cv2
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='')
parser.add_argument('--resize_ratio', dest='resize_ratio', type=float, default=0.4, help='resize iamge')
parser.add_argument('--delay', dest='delay', type=int, default=1, help='interval between two frames')
parser.add_argument('--style_size', dest='style_size', type=int, default=64,
                    help='sylte image size in video')
args = parser.parse_args()


def main():
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('./examples/results/output.mp4', fourcc, 20.0, (768, 1024))

    n_rows, n_cols = 4, 3
    video_path = './examples/{}/fox.mp4'
    style_path = './examples/style/'
    video_file = ['content', 'results/africa', 'results/aquarelle',
                  'results/bango', 'results/chinese_style', 'results/hampson',
                  'results/la_muse', 'results/rain_princess', 'results/the_scream',
                  'results/the_shipwreck_of_the_minotaur', 'results/udnie', 'results/wave']
    img_file = ['africa.jpg', 'aquarelle.jpg',
                'bango.jpg', 'chinese_style.jpg', 'hampson.jpg',
                'la_muse.jpg', 'rain_princess.jpg', 'the_scream.jpg',
                'the_shipwreck_of_the_minotaur.jpg', 'udnie.jpg', 'wave.jpg']

    # initialize video captures & sylte images
    caps, styles = [], []
    for file in video_file:
        caps.append(cv2.VideoCapture(video_path.format(file)))

    for file in img_file:
        styles.append(cv2.imread(os.path.join(style_path, file)))

    cv2.namedWindow('Show')
    cv2.moveWindow('Show', 0, 0)
    while True:
        # read frames
        frames = []
        for idx in range(len(video_file)):
            rest, frame = caps[idx].read()

            if rest is False:
                print('Can not find frame!')
                break
            else:
                # resize original frame
                resized_frame = cv2.resize(frame, (int(frame.shape[0] * args.resize_ratio),
                                                   int(frame.shape[1] * args.resize_ratio)))

                # past style image
                if idx >= 1:
                    img = styles[idx-1]
                    resized_img = cv2.resize(img, (args.style_size, args.style_size))
                    resized_frame[-args.style_size:, 0:args.style_size, :] = resized_img

                frames.append(resized_frame)

        # initialize canvas
        height, width, channel = frames[0].shape
        canvas = np.zeros((n_rows * height, n_cols * width, channel), dtype=np.uint8)

        for row in range(n_rows):
            for col in range(n_cols):
                canvas[row*height:(row+1)*height, col*width:(col+1)*width, :] = frames[row * n_cols + col]

        cv2.imshow('Show', canvas)
        if cv2.waitKey(args.delay) & 0xFF == 27:
            break

        # write the new frame
        out.write(canvas)

    # When everyting done, release the capture
    for idx in range(len(caps)):
        caps[idx].release()
    out.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
