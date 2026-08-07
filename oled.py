import tkinter as tk
from multiprocessing import Process

class OLEDDisplay:
    """
    非阻塞式屏幕显示模块 (基于多进程)
    """
    # 将你的映射字典放在类内部，确保所有方法都能访问
    RESULT_MAP = {
        1: "发现一号人脸", 2: "发现二号人脸", 3: "发现三号人脸", 4: "发现四号人脸", 5: "发现五号人脸",
        6: "发现六号人脸", 7: "发现七号人脸", 8: "发现八号人脸", 9: "发现九号人脸", 10: "发现十号人物",
        11: "发现一号刀具", 12: "发现二号刀具", 13: "发现三号刀具", 14: "发现四号刀具", 15: "发现五号刀具",
        16: "发现六号刀具", 17: "发现七号刀具", 18: "发现八号刀具", 19: "发现九号刀具", 20: "发现十号刀具",
        21: "一号嫌疑人", 22: "二号嫌疑人", 23: "三号嫌疑人", 24: "四号嫌疑人", 25: "五号嫌疑人",
        26: "六号嫌疑人", 27: "七号嫌疑人", 28: "八号嫌疑人", 29: "九号嫌疑人", 30: "十号嫌疑人",
        31: "发现一号可疑刀具", 32: "发现二号可疑刀具", 33: "发现三号可疑刀具", 34: "发现四号可疑刀具", 35: "发现五号可疑刀具",
        36: "发现六号可疑刀具", 37: "发现七号可疑刀具", 38: "发现八号可疑刀具", 39: "发现九号可疑刀具", 40: "发现十号可疑刀具",
    }

    def __init__(self):
        print("[OLED] 屏幕显示模块初始化完成.")
        self.current_process = None

    def _kill_previous(self):
        """如果上一个画面还没关，强制关闭它"""
        if self.current_process and self.current_process.is_alive():
            self.current_process.terminate()
            self.current_process.join()

    def show_text(self, message, duration=2000):
        """显示单行文本"""
        self._kill_previous()
        self.current_process = Process(target=self._ui_single_text, args=(message, duration))
        self.current_process.start()

    def show_detect_result(self, left_res, right_res, duration=3000):
        """显示双侧侦查结果"""
        self._kill_previous()
        self.current_process = Process(target=self._ui_dual_result, args=(left_res, right_res, duration, self.RESULT_MAP))
        self.current_process.start()

    # ================= UI 实现层 =================

    @staticmethod
    def _ui_single_text(message, duration):
        root = tk.Tk()
        root.attributes('-fullscreen', True)
        root.configure(bg='black')
        label = tk.Label(root, text=message, font=("Arial", 80, "bold"), fg='white', bg='black')
        label.pack(expand=True)
        root.after(duration, root.destroy)
        root.mainloop()

    @staticmethod
    def _ui_dual_result(left_res, right_res, duration, result_map):
        # 内部解析逻辑
        def _get_text(res):
            if res == -1: return "未检测到目标"
            return result_map.get(int(res), f"未知目标({res})")

        root = tk.Tk()
        root.attributes('-fullscreen', True)
        root.configure(bg='black')
        root.bind('<Escape>', lambda e: root.destroy())
        root.focus_set()
        
        # 上下分屏
        tk.Label(root, text=f"左侧: {_get_text(left_res)}", font=("Arial", 60), fg='cyan', bg='black').pack(expand=True)
        tk.Label(root, text=f"右侧: {_get_text(right_res)}", font=("Arial", 60), fg='yellow', bg='black').pack(expand=True)
        
        root.after(duration, root.destroy)
        root.mainloop()