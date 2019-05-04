# -*- coding: utf-8 -*-
"""
@Date: Created on 2019/5/4
@Author: Haojun Gao
@Description: 在训练模型时，在迭代中，用显示进度条的方式实时显示训练进度，并且不需要被刷屏。
"""

import sys


class ProgressBar:

    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.current_step = 0
        self.progress_width = 50

    def update(self, step=None):
        self.current_step = step

        num_pass = int(self.current_step * self.progress_width / self.max_steps) + 1
        num_rest = self.progress_width - num_pass
        percent = (self.current_step + 1) * 100.0 / self.max_steps
        progress_bar = '[' + '■' * (num_pass - 1) + '▶' + '-' * num_rest + ']'
        progress_bar += '%.2f' % percent + '%'
        if self.current_step < self.max_steps - 1:
            progress_bar += '\r'
        else:
            progress_bar += '\n'

        sys.stdout.write(progress_bar)
        sys.stdout.flush()

        if self.current_step >= self.max_steps:
            self.current_step = 0
