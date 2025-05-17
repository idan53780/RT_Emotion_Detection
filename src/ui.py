import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

def create_start_screen(app):
    # Clear the root window
    for widget in app.root.winfo_children():
        widget.destroy()
    
    # Main frame
    main_frame = tk.Frame(app.root, bg="#2c3e50")
    main_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
    
    # Title with logo
    title_frame = tk.Frame(main_frame, bg="#2c3e50")
    title_frame.pack(pady=20)
    
    if app.logo_photo:
        logo_label = tk.Label(title_frame, image=app.logo_photo, bg="#2c3e50")
        logo_label.pack(side=tk.LEFT, padx=10)
    
    title_label = tk.Label(
        title_frame, 
        text="EmotionLens", 
        font=("Helvetica", 36, "bold"), 
        fg="#ecf0f1", 
        bg="#2c3e50"
    )
    title_label.pack(side=tk.LEFT)
    
    subtitle = tk.Label(
        main_frame, 
        text="Real-Time Emotion Detection", 
        font=("Helvetica", 18), 
        fg="#bdc3c7", 
        bg="#2c3e50"
    )
    subtitle.pack(pady=(0, 40))
    
    # Buttons frame
    button_frame = tk.Frame(main_frame, bg="#2c3e50")
    button_frame.pack(pady=20)
    
    # Button style
    button_style = {
        "font": ("Helvetica", 14),
        "bg": "#3498db",
        "fg": "white",
        "activebackground": "#2980b9",
        "activeforeground": "white",
        "width": 20,
        "height": 2,
        "borderwidth": 0,
        "cursor": "hand2"
    }
    
    # Start with webcam button
    webcam_btn = tk.Button(
        button_frame, 
        text="Start with Webcam", 
        command=lambda: app.start_detection("webcam"),
        **button_style
    )
    webcam_btn.pack(pady=10)
    
    # Start with video file button
    video_btn = tk.Button(
        button_frame, 
        text="Start with Video File", 
        command=lambda: app.start_detection("video"),
        **button_style
    )
    video_btn.pack(pady=10)
    
    # Load image button
    image_btn = tk.Button(
        button_frame, 
        text="Analyze Image", 
        command=lambda: app.start_detection("image"),
        **button_style
    )
    image_btn.pack(pady=10)
    
    # Settings button
    settings_btn = tk.Button(
        button_frame, 
        text="Settings", 
        command=app.open_settings,
        **button_style
    )
    settings_btn.pack(pady=10)
    
    # Exit button
    exit_btn = tk.Button(
        button_frame, 
        text="Exit", 
        command=app.root.quit,
        **button_style
    )
    exit_btn.pack(pady=10)
    
    # Version and author
    footer = tk.Label(
        main_frame, 
        text="v1.0.0 | Created by Idan53780", 
        font=("Helvetica", 10), 
        fg="#95a5a6", 
        bg="#2c3e50"
    )
    footer.pack(side=tk.BOTTOM, pady=10)

def open_settings(app):
    settings_window = tk.Toplevel(app.root)
    settings_window.title("EmotionLens Settings")
    settings_window.geometry("600x600")
    settings_window.configure(bg="#2c3e50")
    settings_window.grab_set()  # Modal window
    
    # Settings frame
    settings_frame = tk.Frame(settings_window, bg="#2c3e50", padx=20, pady=20)
    settings_frame.pack(fill=tk.BOTH, expand=True)
    
    # Title
    title = tk.Label(
        settings_frame, 
        text="Settings", 
        font=("Helvetica", 24, "bold"), 
        fg="#ecf0f1", 
        bg="#2c3e50"
    )
    title.pack(pady=(0, 20))
    
    # Face Detection Model Selection
    face_detector_frame = tk.LabelFrame(
        settings_frame, 
        text="Face Detection Model", 
        font=("Helvetica", 12), 
        fg="#ecf0f1", 
        bg="#2c3e50",
        padx=10, 
        pady=10
    )
    face_detector_frame.pack(fill=tk.X, pady=10)
    
    detector_var = tk.StringVar(value=app.current_detector)
    
    for detector in ["Haar Cascade", "DNN"]:
        rb = tk.Radiobutton(
            face_detector_frame,
            text=detector,
            variable=detector_var,
            value=detector,
            bg="#2c3e50",
            fg="#ecf0f1",
            selectcolor="#2c3e50",
            activebackground="#2c3e50",
            activeforeground="#ecf0f1",
            font=("Helvetica", 12)
        )
        rb.pack(anchor=tk.W)
    
    # Emotion Model Selection
    emotion_model_frame = tk.LabelFrame(
        settings_frame, 
        text="Emotion Recognition Model", 
        font=("Helvetica", 12), 
        fg="#ecf0f1", 
        bg="#2c3e50",
        padx=10, 
        pady=10
    )
    emotion_model_frame.pack(fill=tk.X, pady=10)
    
    emotion_model_var = tk.StringVar(value=app.current_emotion_model)
    emotion_model_combo = ttk.Combobox(
        emotion_model_frame,
        textvariable=emotion_model_var,
        values=app.emotion_models,
        font=("Helvetica", 12),
        state="readonly"
    )
    emotion_model_combo.pack(fill=tk.X, pady=5)
    
    # Confidence Threshold
    threshold_frame = tk.LabelFrame(
        settings_frame, 
        text="Confidence Threshold", 
        font=("Helvetica", 12), 
        fg="#ecf0f1", 
        bg="#2c3e50",
        padx=10, 
        pady=10
    )
    threshold_frame.pack(fill=tk.X, pady=10)
    
    threshold_var = tk.DoubleVar(value=app.confidence_threshold)
    threshold_scale = tk.Scale(
        threshold_frame,
        from_=0.0,
        to=1.0,
        resolution=0.05,
        orient=tk.HORIZONTAL,
        variable=threshold_var,
        bg="#2c3e50",
        fg="#ecf0f1",
        highlightthickness=0,
        troughcolor="#34495e",
        activebackground="#3498db",
        font=("Helvetica", 12)
    )
    threshold_scale.pack(fill=tk.X)
    
    # History length for temporal smoothing
    history_frame = tk.LabelFrame(
        settings_frame, 
        text="Temporal Smoothing Frames", 
        font=("Helvetica", 12), 
        fg="#ecf0f1", 
        bg="#2c3e50",
        padx=10, 
        pady=10
    )
    history_frame.pack(fill=tk.X, pady=10)
    
    history_var = tk.IntVar(value=app.history_length)
    history_scale = tk.Scale(
        history_frame,
        from_=1,
        to=10,
        resolution=1,
        orient=tk.HORIZONTAL,
        variable=history_var,
        bg="#2c3e50",
        fg="#ecf0f1",
        highlightthickness=0,
        troughcolor="#34495e",
        activebackground="#3498db",
        font=("Helvetica", 12)
    )
    history_scale.pack(fill=tk.X)
    
    # Save button
    save_btn = tk.Button(
        settings_frame,
        text="Save Settings",
        command=lambda: app.save_settings(
            detector_var.get(),
            emotion_model_var.get(),
            threshold_var.get(),
            history_var.get(),
            settings_window
        ),
        font=("Helvetica", 14),
        bg="#2ecc71",
        fg="white",
        activebackground="#27ae60",
        activeforeground="white",
        width=15,
        height=1,
        borderwidth=0,
        cursor="hand2"
    )
    save_btn.pack(pady=20)

def create_app_interface(app):
    # Clear the root window
    for widget in app.root.winfo_children():
        widget.destroy()
    
    # Main frame
    main_frame = tk.Frame(app.root, bg="#2c3e50")
    main_frame.pack(expand=True, fill=tk.BOTH)
    
    # Top control panel
    control_panel = tk.Frame(main_frame, bg="#34495e", height=50)
    control_panel.pack(fill=tk.X)
    
    if app.logo_photo:
        logo_label = tk.Label(control_panel, image=app.logo_photo, bg="#34495e")
        logo_label.pack(side=tk.LEFT, padx=10, pady=5)
    
    title_label = tk.Label(
        control_panel, 
        text="EmotionLens", 
        font=("Helvetica", 16, "bold"), 
        fg="#ecf0f1", 
        bg="#34495e"
    )
    title_label.pack(side=tk.LEFT, padx=5)
    
    # Control buttons
    btn_frame = tk.Frame(control_panel, bg="#34495e")
    btn_frame.pack(side=tk.RIGHT, padx=10)
    
    # Screenshot button
    screenshot_btn = tk.Button(
        btn_frame,
        text="Screenshot",
        command=app.take_screenshot,
        font=("Helvetica", 12),
        bg="#3498db",
        fg="white",
        activebackground="#2980b9",
        activeforeground="white",
        padx=10,
        borderwidth=0
    )
    screenshot_btn.pack(side=tk.LEFT, padx=5)
    
    # Return to menu button
    menu_btn = tk.Button(
        btn_frame,
        text="Back to Menu",
        command=app.stop_and_return,
        font=("Helvetica", 12),
        bg="#e74c3c",
        fg="white",
        activebackground="#c0392b",
        activeforeground="white",
        padx=10,
        borderwidth=0
    )
    menu_btn.pack(side=tk.LEFT, padx=5)
    
    # Video display frame
    app.video_frame = tk.Label(main_frame, bg="black")
    app.video_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
    
    # Status bar
    status_frame = tk.Frame(main_frame, bg="#34495e", height=30)
    status_frame.pack(fill=tk.X, side=tk.BOTTOM)
    
    app.status_label = tk.Label(
        status_frame, 
        text="Ready", 
        font=("Helvetica", 10), 
        fg="#ecf0f1", 
        bg="#34495e",
        anchor=tk.W
    )
    app.status_label.pack(side=tk.LEFT, padx=10, pady=5)
    
    # Version info
    version_label = tk.Label(
        status_frame, 
        text="v1.0.0", 
        font=("Helvetica", 10), 
        fg="#ecf0f1", 
        bg="#34495e",
        anchor=tk.E
    )
    version_label.pack(side=tk.RIGHT, padx=10, pady=5)
    

def create_image_analysis_interface(app):
    # Clear the root window
    for widget in app.root.winfo_children():
        widget.destroy()
    
    # Main frame
    main_frame = tk.Frame(app.root, bg="#2c3e50")
    main_frame.pack(expand=True, fill=tk.BOTH)
    
    # Top control panel
    control_panel = tk.Frame(main_frame, bg="#34495e", height=50)
    control_panel.pack(fill=tk.X)
    
    if app.logo_photo:
        logo_label = tk.Label(control_panel, image=app.logo_photo, bg="#34495e")
        logo_label.pack(side=tk.LEFT, padx=10, pady=5)
    
    title_label = tk.Label(
        control_panel, 
        text="EmotionLens - Image Analysis", 
        font=("Helvetica", 16, "bold"), 
        fg="#ecf0f1", 
        bg="#34495e"
    )
    title_label.pack(side=tk.LEFT, padx=5)
    
    # Control buttons
    btn_frame = tk.Frame(control_panel, bg="#34495e")
    btn_frame.pack(side=tk.RIGHT, padx=10)
    
    # Save button
    save_btn = tk.Button(
        btn_frame,
        text="Save Result",
        command=app.save_analysis,
        font=("Helvetica", 12),
        bg="#3498db",
        fg="white",
        activebackground="#2980b9",
        activeforeground="white",
        padx=10,
        borderwidth=0
    )
    save_btn.pack(side=tk.LEFT, padx=5)
    
    # Return to menu button
    menu_btn = tk.Button(
        btn_frame,
        text="Back to Menu",
        command=app.create_start_screen,
        font=("Helvetica", 12),
        bg="#e74c3c",
        fg="white",
        activebackground="#c0392b",
        activeforeground="white",
        padx=10,
        borderwidth=0
    )
    menu_btn.pack(side=tk.LEFT, padx=5)
    
    # Image display frame
    app.video_frame = tk.Label(main_frame, bg="black")
    app.video_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
    
    # Status bar
    status_frame = tk.Frame(main_frame, bg="#34495e", height=30)
    status_frame.pack(fill=tk.X, side=tk.BOTTOM)
    
    app.status_label = tk.Label(
        status_frame, 
        text="Analyzing image...", 
        font=("Helvetica", 10), 
        fg="#ecf0f1", 
        bg="#34495e",
        anchor=tk.W
    )
    app.status_label.pack(side=tk.LEFT, padx=10, pady=5)
   