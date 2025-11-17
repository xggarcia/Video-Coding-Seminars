class Color_translator:

    @staticmethod
    def RGB_YUV_translator(value_1, value_2, value_3, mode='RGB_to_YUV'):
        if mode == 'YUV_to_RGB':
            Y, U, V = value_1, value_2, value_3
            R = round(Y + 1.13983 * V, 0)
            G = round(Y - 0.39465 * U - 0.58060 * V, 0)
            B = round(Y + 2.03211 * U, 0)
            return (R, G, B)
        else:  # Default mode is 'RGB_to_YUV'
            R, G, B = value_1, value_2, value_3
            Y = round(0.299 * R + 0.587 * G + 0.114 * B, 0)
            U = round(-0.14713 * R - 0.28886 * G + 0.436 * B, 0)
            V = round(0.615 * R - 0.51499 * G - 0.10001 * B, 0)
            return (Y, U, V)

def main():
    print("RGB <-> YUV Translator")
    print("----------------------")

    # You could also read from a text file or argv; here we use simple input().
    try:
        rgb_str = input("Enter R G B values (0-255) separated by spaces (e.g. '128 64 32'): ")
        r_str, g_str, b_str = rgb_str.strip().split()
        r, g, b = int(r_str), int(g_str), int(b_str)

        translator = Color_translator()

        # RGB -> YUV
        y, u, v = translator.RGB_YUV_translator(r, g, b, mode='RGB_to_YUV')
        print(f"\nRGB ({r}, {g}, {b}) ->")
        print(f"YUV (Y={y:.2f}, U={u:.2f}, V={v:.2f})")

        # YUV -> RGB (inverse)
        r2, g2, b2 = translator.RGB_YUV_translator(y, u, v, mode='YUV_to_RGB')
        print(f"\nBack to RGB from YUV ->")
        print(f"RGB ({r2}, {g2}, {b2})")

    except ValueError:
        print("Error: please enter exactly three integer values for R, G and B.")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()