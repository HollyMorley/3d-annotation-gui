import tkinter as tk

from annotation_tool.gui.app import MainTool


def main():
    root = tk.Tk()
    MainTool(root)
    root.mainloop()


if __name__ == "__main__":
    main()
