from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from typing import List, Optional, Tuple
import matplotlib
from omegaconf import DictConfig
import os
matplotlib.use('Agg')  # Set backend to Agg

@dataclass
class MouseVisualizer:
    """마우스 트래킹 시각화 클래스"""
    
    def __init__(self, frame_width: int, frame_height: int,
                 mouse_colors: List[str], plot_pairs: List[Tuple[int, int]]):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.mouse_colors = mouse_colors
        self.plot_pairs = plot_pairs
        
    def set_figax(self):
        """Figure와 Axes 설정"""
        fig = plt.figure(figsize=(8, 8))
        img = np.zeros((self.frame_height, self.frame_width, 3))
        ax = fig.add_subplot(111)
        ax.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        return fig, ax
        
    def plot_mouse(self, ax, pose: np.ndarray, color: str) -> None:
        """단일 마우스 포즈 플롯"""
        for j in range(10):
            ax.plot(pose[j, 0], pose[j, 1], 'o', color=color, markersize=3)
            
        for pair in self.plot_pairs:
            line_to_plot = pose[pair, :]
            ax.plot(line_to_plot[:, 0], line_to_plot[:, 1], 
                   color=color, linewidth=1)
                   
    def animate_sequence(self, video_name: str, seq: np.ndarray,
                        start_frame: int = 0, stop_frame: int = 100,
                        skip: int = 0, annotation_sequence: Optional[np.ndarray] = None,
                        cfg: DictConfig = None
                        ) -> animation.FuncAnimation:
        """시퀀스 애니메이션 생성"""
        image_list = []
        
        counter = 0
        if skip:
            anim_range = range(start_frame, stop_frame, skip)
        else:
            anim_range = range(start_frame, stop_frame)
        
        for j in anim_range:
            if counter % 20 == 0:
                print("Processing frame ", j)
            fig, ax = self.set_figax()
            
            # Plot each mouse
            for m, color in enumerate(self.mouse_colors):
                self.plot_mouse(ax, seq[j, m, :, :], color=color)
            
            if annotation_sequence is not None:
                annot = annotation_sequence[j]
                plt.text(50, -20, annot, fontsize=16,
                        bbox=dict(facecolor='white', alpha=0.5))
            
            ax.set_title(f"{video_name}\n frame {j:03d}.png")
            ax.axis('off')
            fig.tight_layout(pad=0)
            ax.margins(0)
            
            # Convert plot to image
            fig.canvas.draw()
            # Get the RGBA buffer from the figure
            w, h = fig.canvas.get_width_height()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            # Reshape it to a NumPy array of the right shape
            image_from_plot = buf.reshape(h, w, 4)
            # Convert RGBA to RGB
            image_from_plot = image_from_plot[:,:,:3]
            
            image_list.append(image_from_plot)
            plt.close()
            counter += 1
            
        # Create animation
        fig = plt.figure(figsize=(8,8))
        plt.axis('off')
        im = plt.imshow(image_list[0])
        
        def animate(k):
            im.set_array(image_list[k])
            return im,
            
        ani = animation.FuncAnimation(fig, animate, frames=len(image_list), blit=True)

        os.makedirs(cfg.data.paths.visualizations_dir, exist_ok=True)
        # Save animation
        animation_path = cfg.data.paths.visualizations_dir / f"{video_name}_animation.gif"
        ani.save(animation_path, writer='pillow')
        print(f"\nAnimation saved to: {animation_path}")
        