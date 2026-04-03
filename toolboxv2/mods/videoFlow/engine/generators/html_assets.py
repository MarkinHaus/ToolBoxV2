# toolboxv2/mods/videoFlow/engine/generators/html_assets.py


def get_camera_specific_css(camera_style: str) -> str:
    """Generate camera-specific CSS effects"""

    camera_effects = {
        # Black & White styles
        "black_white_classic": "filter: grayscale(100%) contrast(1.2);",
        "film_noir": "filter: grayscale(100%) contrast(1.5) brightness(0.8);",
        "high_contrast_bw": "filter: grayscale(100%) contrast(2.0);",
        "vintage_bw": "filter: grayscale(100%) sepia(0.3) contrast(1.1);",

        # Bright/Colorful effects
        "neon_cyberpunk": "filter: saturate(1.8) contrast(1.3) hue-rotate(10deg);",
        "vaporwave": "filter: saturate(1.5) hue-rotate(270deg) contrast(1.2);",
        "psychedelic": "filter: saturate(2.0) hue-rotate(180deg) contrast(1.4);",
        "rainbow_bright": "filter: saturate(1.8) brightness(1.1) contrast(1.2);",
        "candy_colors": "filter: saturate(1.6) brightness(1.05) contrast(1.1);",
        "miami_vice": "filter: saturate(1.4) hue-rotate(315deg) contrast(1.2);",

        # Special effects
        "glitch_art": "filter: saturate(1.5) contrast(1.3); animation: glitch 2s infinite;",
        "holographic": "filter: saturate(1.4) brightness(1.1); animation: hologram 3s infinite;",
        "thermal_camera": "filter: hue-rotate(200deg) saturate(1.8) contrast(1.3);",
        "infrared": "filter: hue-rotate(270deg) saturate(1.2) contrast(1.4);",

        # Vintage effects
        "polaroid_vintage": "filter: sepia(0.4) saturate(0.8) contrast(0.9);",
        "film_35mm": "filter: sepia(0.2) saturate(1.1) contrast(1.05);",

        # Art styles
        "impressionist": "filter: blur(0.5px) saturate(1.2) contrast(0.9);",
        "expressionist": "filter: saturate(1.6) contrast(1.4) brightness(0.95);",
    }

    return camera_effects.get(camera_style.lower().replace(" ", "_").replace("&", ""), "")

def get_dark_mode_css(story_data, colors: dict) -> str:
    """Generate DARK MODE CSS for all themes with IMAGE MODAL functionality"""

    return f"""
    :root {{
        --bg-color: {colors['bg']};
        --surface-color: {colors['surface']};
        --primary-color: {colors['primary']};
        --secondary-color: {colors['secondary']};
        --accent-color: {colors['accent']};
        --glass-bg: rgba(255,255,255,0.05);
        --glass-border: rgba(255,255,255,0.1);
    }}

    * {{
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }}

    @keyframes glow {{
        from {{ box-shadow: 0 20px 40px rgba(0,0,0,0.6); }}
        to {{ box-shadow: 0 20px 40px var(--accent-color)33; }}
    }}

    body {{
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        line-height: 1.6;
        color: var(--primary-color);
        background: var(--bg-color);
        overflow-x: hidden;
    }}

    .main-container {{
        max-width: 100%;
        margin: 0 auto;
        background: var(--bg-color);
        min-height: 100vh;
    }}

    /* IMAGE MODAL STYLES */
    .image-modal {{
        display: none;
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.95);
        z-index: 10000;
        backdrop-filter: blur(10px);
        opacity: 0;
        transition: opacity 0.3s ease;
    }}

    .image-modal.show {{
        display: flex;
        opacity: 1;
        align-items: center;
        justify-content: center;
        animation: modalFadeIn 0.3s ease-out;
    }}

    .modal-content {{
        position: relative;
        max-width: 95%;
        max-height: 95%;
        background: var(--surface-color);
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.8);
        border: 1px solid var(--glass-border);
        transform: scale(0.8);
        transition: transform 0.3s ease;
    }}

    .image-modal.show .modal-content {{
        transform: scale(1);
    }}

    .modal-image {{
        width: 100%;
        height: auto;
        max-height: 80vh;
        object-fit: contain;
        display: block;
        background: var(--bg-color);
    }}

    .modal-info {{
        padding: 1.5rem;
        background: var(--surface-color);
        border-top: 1px solid var(--glass-border);
    }}

    .modal-title {{
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--accent-color);
        margin-bottom: 0.5rem;
    }}

    .modal-description {{
        color: var(--secondary-color);
        font-size: 1rem;
        line-height: 1.6;
        margin-bottom: 1rem;
    }}

    .modal-details {{
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        font-size: 0.9rem;
        color: var(--secondary-color);
    }}

    .modal-detail {{
        background: var(--glass-bg);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        border: 1px solid var(--glass-border);
    }}

    .modal-close {{
        position: absolute;
        top: 15px;
        right: 15px;
        background: transparent;
        color: white;
        border: none;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        font-size: 1.5rem;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.3s;
        z-index: 10001;
    }}

    .modal-close:hover {{
        background: var(--primary-color);
        color: var(--bg-color);
        transform: scale(1.1);
    }}

    .modal-navigation {{
        position: absolute;
        top: 50%;
        transform: translateY(-50%);
        background: transparent;
        color: white;
        border: none;
        width: 50px;
        height: 50px;
        border-radius: 50%;
        font-size: 1.5rem;
        cursor: pointer;
        transition: all 0.3s;
        z-index: 10001;
    }}

    .modal-navigation:hover {{
        background: var(--primary-color);
        color: var(--bg-color);
        transform: translateY(-50%) scale(1.1);
    }}

    .modal-prev {{
        left: 20px;
    }}

    .modal-next {{
        right: 20px;
    }}

    @keyframes modalFadeIn {{
        from {{
            opacity: 0;
            backdrop-filter: blur(0px);
        }}
        to {{
            opacity: 1;
            backdrop-filter: blur(10px);
        }}
    }}

    /* Clickable Images */
    .clickable-image {{
        cursor: pointer;
        transition: all 0.3s ease;
        position: relative;
    }}

    .clickable-image::before {{
        content: '🔍';
        position: absolute;
        top: 10px;
        right: 10px;
        background: rgba(0, 0, 0, 0.7);
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 1.2rem;
        opacity: 0;
        transition: opacity 0.3s;
        z-index: 10;
    }}

    .clickable-image:hover::before {{
        opacity: 1;
    }}

    .clickable-image:hover {{
        transform: scale(1.05);
        filter: brightness(1.1);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }}

    /* Header Section - DARK */
    .hero-section {{
        position: relative;
        min-height: 100vh;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        background: linear-gradient(135deg, var(--bg-color) 0%, var(--surface-color) 100%);
        color: var(--primary-color);
        text-align: center;
        padding: 2rem;
    }}

    .hero-title-image {{
        max-width: 90%;
        max-height: 50vh;
        object-fit: contain;
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.6);
        margin-bottom: 2rem;
        border: 2px solid var(--glass-border);
        animation: glow 2s ease-in-out infinite alternate;
    }}


    .hero-title {{
        font-size: clamp(2.5rem, 6vw, 5rem);
        font-weight: 900;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.8);
        margin-bottom: 1rem;
        background: linear-gradient(45deg, var(--primary-color), var(--accent-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}

    .hero-subtitle {{
        font-size: clamp(1.2rem, 3vw, 2rem);
        opacity: 0.9;
        margin-bottom: 2rem;
        color: var(--secondary-color);
        word-break: break-word;
        max-width: 90%;
    }}

    .hero-video {{
        width: 100%;
        max-width: 900px;
        border-radius: 15px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.5);
        border: 1px solid var(--glass-border);
    }}

    /* FIXED Audio Player Popup */
    .audio-player-floating {{
        position: fixed;
        top: 20px;
        right: 20px;
        background: var(--surface-color);
        backdrop-filter: blur(20px);
        color: var(--primary-color);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        z-index: 1000;
        min-width: 280px;
        border: 1px solid var(--glass-border);
        transform: translateX(100%);
        transition: transform 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55);
    }}

    .audio-player-floating.show {{
        transform: translateX(0);
    }}

    .audio-player-floating.hide {{
        transform: translateX(90%);
    }}

    .audio-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }}

    .close-audio-btn {{
        background: none;
        border: none;
        color: var(--primary-color);
        font-size: 1.5rem;
        cursor: pointer;
        padding: 5px;
        border-radius: 50%;
        transition: background 0.3s;
    }}

    .close-audio-btn:hover {{
        background: var(--glass-bg);
    }}

    .audio-controls {{
        display: flex;
        gap: 10px;
        margin-top: 15px;
        flex-wrap: wrap;
    }}

    .audio-btn {{
        background: var(--accent-color);
        color: white;
        border: none;
        padding: 10px 15px;
        border-radius: 8px;
        cursor: pointer;
        font-size: 0.9rem;
        font-weight: 600;
        transition: all 0.3s;
        flex: 1;
        min-width: 80px;
    }}

    .audio-btn:hover {{
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }}

    .audio-progress {{
        margin-top: 10px;
        font-size: 0.85rem;
        color: var(--secondary-color);
        text-align: center;
    }}

    /* Section Styling - DARK */
    .content-section {{
        padding: 5rem 2rem;
        max-width: 1400px;
        margin: 0 auto;
        background: var(--bg-color);
    }}

    .section-title {{
        font-size: clamp(2rem, 5vw, 3.5rem);
        color: var(--primary-color);
        margin-bottom: 3rem;
        text-align: center;
        position: relative;
        font-weight: 700;
    }}

    .section-title::after {{
        content: '';
        position: absolute;
        bottom: -15px;
        left: 50%;
        transform: translateX(-50%);
        width: 100px;
        height: 4px;
        background: linear-gradient(90deg, var(--accent-color), transparent);
        border-radius: 2px;
    }}

    /* Media Grid - DARK */
    .media-showcase {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
        gap: 2.5rem;
        margin-top: 3rem;
    }}

    .media-card {{
        background: var(--surface-color);
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0,0,0,0.4);
        transition: all 0.4s ease;
        border: 1px solid var(--glass-border);
    }}

    .media-card-gold {{
        background: darkgoldenrod;
        border-radius: 20px;
        text-color: black;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0,0,0,0.4);
        transition: all 0.4s ease;
        border: 1px solid var(--glass-border);
        color: black !important;
    }}

    .media-card:hover {{
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 20px 50px rgba(0,0,0,0.6);
    }}

    .media-card img {{
        width: 100%;
        height: 280px;
        object-fit: cover;
        transition: transform 0.4s;
    }}

    .media-card:hover img {{
        transform: scale(1.1);
    }}

    .media-card video {{
        width: 100%;
        max-height: 350px;
        background: var(--bg-color);
    }}

    .media-info {{
        padding: 1.5rem;
        color: var(--primary-color);
    }}


    .media-title {{
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        word-break: break-word;
        hyphens: auto;
        color: var(--accent-color);
    }}

    .media-description {{
        color: var(--secondary-color);
        font-size: 0.95rem;
        line-height: 1.6;
    }}

    /* Story Scene Styling - DARK */
    .story-scene {{
        margin-bottom: 5rem;
        padding: 3rem;
        background: var(--surface-color);
        border-radius: 25px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        border: 1px solid var(--glass-border);
        position: relative;
    }}

    .story-scene-glod {{
        margin-bottom: 5rem;
        padding: 3rem;
        background: darkgoldenrod;
        border-radius: 25px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        border: 1px solid var(--glass-border);
        position: relative;
    }}

    .scene-title {{
        font-size: 2.2rem;
        color: var(--accent-color);
        margin-bottom: 1.5rem;
        font-weight: 700;
        border-left: 5px solid var(--accent-color);
        padding-left: 2rem;
    }}

    .narrator-content {{
        font-size: 1.3rem;
        line-height: 1.8;
        color: var(--primary-color);
        margin-bottom: 2.5rem;
        font-style: italic;
        padding: 2rem;
        background: var(--glass-bg);
        border-radius: 15px;
        border: 1px solid var(--glass-border);
    }}

    .dialogue-container {{
        margin: 2rem 0;
        padding: 1.5rem;
        background: var(--glass-bg);
        border-radius: 15px;
        border-left: 4px solid var(--accent-color);
    }}

    .character-speaker {{
        font-weight: 700;
        color: var(--accent-color);
        font-size: 1.1rem;
        margin-bottom: 0.8rem;
    }}

    .dialogue-speech {{
        color: var(--primary-color);
        line-height: 1.7;
        font-size: 1.05rem;
    }}

    /* Production Info Grid - DARK */
    .info-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 2rem;
        margin-top: 3rem;
    }}

    .info-card {{
        background: var(--surface-color);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.3);
        border: 1px solid var(--glass-border);
        transition: transform 0.3s;
    }}

    .info-card:hover {{
        transform: translateY(-5px);
    }}

    .info-card h3 {{
        color: var(--accent-color);
        margin-bottom: 1rem;
        font-size: 1.3rem;
    }}

    .info-card p {{
        color: var(--secondary-color);
        line-height: 1.6;
        margin-bottom: 0.5rem;
    }}

    /* End Card - DARK */
    .finale-section {{
        min-height: 100vh;
        background: linear-gradient(135deg, var(--bg-color), var(--surface-color));
        color: var(--primary-color);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        padding: 3rem;
    }}

    .finale-title {{
        font-size: clamp(3rem, 8vw, 6rem);
        margin-bottom: 2rem;
        font-weight: 900;
        text-shadow: 3px 3px 10px rgba(0,0,0,0.7);
        background: linear-gradient(45deg, var(--primary-color), var(--accent-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}

    /* Scroll Animations */
    .fade-in-up {{
        opacity: 0;
        transform: translateY(50px);
        transition: all 0.8s cubic-bezier(0.25, 0.46, 0.45, 0.94);
    }}

    .fade-in-up.visible {{
        opacity: 1;
        transform: translateY(0);
    }}

    /* Transitions */
    .transitions-section {{
        border-top: 1px solid var(--glass-border);
        border-bottom: 1px solid var(--glass-border);
        padding: 1.5rem 0;
        background: var(--glass-bg);
        border-radius: 10px;
    }}

    .transition-card {{
        border: 2px solid var(--accent-color);
        background: linear-gradient(135deg, var(--surface-color), var(--glass-bg));
    }}

    .transition-card:hover {{
        transform: translateY(-5px) scale(1.03);
        border-color: var(--primary-color);
        box-shadow: 0 15px 40px rgba(0,0,0,0.5);
    }}

    .transition-card img {{
        height: 200px;
        object-fit: cover;
        filter: sepia(0.1) saturate(1.1);
    }}

    /* PDF Section Styling */
    .pdf-showcase {{
        display: flex;
        flex-direction: column;
        gap: 3rem;
        margin-top: 3rem;
    }}

    .pdf-container {{
        background: var(--surface-color);
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0,0,0,0.4);
        border: 1px solid var(--glass-border);
    }}

    .pdf-header {{
        padding: 2rem;
        background: linear-gradient(135deg, var(--surface-color), var(--glass-bg));
        border-bottom: 1px solid var(--glass-border);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 1rem;
    }}

    .pdf-title {{
        color: var(--accent-color);
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0;
    }}

    .pdf-meta {{
        display: flex;
        align-items: center;
        gap: 1rem;
        color: var(--secondary-color);
        flex-wrap: wrap;
    }}

    .download-btn {{
        background: var(--accent-color);
        color: white;
        border: none;
        padding: 12px 20px;
        border-radius: 10px;
        cursor: pointer;
        font-weight: 600;
        transition: all 0.3s;
        text-decoration: none;
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }}

    .download-btn:hover {{
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        filter: brightness(1.1);
    }}

    .pdf-viewer-container {{
        position: relative;
        background: var(--bg-color);
    }}

    .pdf-viewer {{
        width: 100%;
        height: 800px;
        border: none;
        background: white;
        display: block;
    }}

    .pdf-controls {{
        padding: 1.5rem;
        background: var(--surface-color);
        display: flex;
        justify-content: center;
        gap: 1rem;
        flex-wrap: wrap;
        border-top: 1px solid var(--glass-border);
    }}

    .pdf-btn {{
        background: var(--glass-bg);
        color: var(--primary-color);
        border: 1px solid var(--glass-border);
        padding: 10px 16px;
        border-radius: 8px;
        cursor: pointer;
        text-decoration: none;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: 0.9rem;
        transition: all 0.3s;
    }}

    .pdf-btn:hover {{
        background: var(--accent-color);
        color: white;
        transform: translateY(-1px);
    }}

    /* Responsive Design */
    @media (max-width: 768px) {{
        .content-section {{
            padding: 3rem 1rem;
        }}

        .media-showcase {{
            grid-template-columns: 1fr;
            gap: 1.5rem;
        }}

        .info-grid {{
            grid-template-columns: 1fr;
        }}

        .story-scene {{
            padding: 1.5rem;
            margin-bottom: 3rem;
        }}

        .audio-player-floating {{
            right: 10px;
            top: 10px;
            min-width: 250px;
            padding: 1rem;
        }}

        .modal-navigation {{
            width: 40px;
            height: 40px;
            font-size: 1.2rem;
        }}

        .modal-prev {{
            left: 10px;
        }}

        .modal-next {{
            right: 10px;
        }}

        .modal-info {{
            padding: 1rem;
        }}

        .pdf-viewer {{
            height: 600px;
        }}

        .pdf-header {{
            flex-direction: column;
            align-items: flex-start;
        }}

        .pdf-controls {{
            flex-direction: column;
        }}

        .pdf-btn, .download-btn {{
            width: 100%;
            justify-content: center;
        }}
    }}

    /* Loading Animation */
    .loading-shimmer {{
        background: linear-gradient(90deg, transparent, var(--glass-bg), transparent);
        background-size: 200% 100%;
        animation: shimmer 1.5s infinite;
    }}

    @keyframes shimmer {{
        0% {{ background-position: -200% 0; }}
        100% {{ background-position: 200% 0; }}
    }}

    .hero-title-image, .media-card img {{
        {get_camera_specific_css(story_data.style_preset.camera_style.value)}
    }}

    /* Special animations for camera effects */
    @keyframes glitch {{
        0% {{ transform: translate(0); }}
        20% {{ transform: translate(-2px, 2px); }}
        40% {{ transform: translate(-2px, -2px); }}
        60% {{ transform: translate(2px, 2px); }}
        80% {{ transform: translate(2px, -2px); }}
        100% {{ transform: translate(0); }}
    }}

    @keyframes hologram {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.8; filter: hue-rotate(0deg); }}
    }}

    /* Notification Animations */
    @keyframes slideInDown {{
        from {{
            transform: translate(-50%, -100%);
            opacity: 0;
        }}
        to {{
            transform: translate(-50%, 0);
            opacity: 1;
        }}
    }}

    @keyframes slideOutUp {{
        from {{
            transform: translate(-50%, 0);
            opacity: 1;
        }}
        to {{
            transform: translate(-50%, -100%);
            opacity: 0;
        }}
    }}
    """

def get_fixed_javascript() -> str:
    """Generate FIXED JavaScript with PROPER modal cleanup to prevent freezing"""

    return """
    // Global variables
    let audioPlayer = null;
    let isAudioPlaying = false;
    let audioPlayerVisible = false;
    let currentImageModal = null;
    let allImages = [];
    let currentImageIndex = 0;
    let scrollPosition = 0;

    // Initialize everything when page loads
    document.addEventListener('DOMContentLoaded', function() {
        console.log('Page loaded, initializing...');

        // Initialize audio player
        initializeAudioPlayer();

        // Initialize scroll animations
        initializeScrollAnimations();

        // Initialize image modal system
        initializeImageModals();

        // Show audio player after 3 seconds
        setTimeout(() => {
            showAudioPlayer();
        }, 3000);
    });

    // IMAGE MODAL FUNCTIONALITY
    function initializeImageModals() {
        console.log('Initializing image modals...');

        // Collect all clickable images
        allImages = Array.from(document.querySelectorAll('.clickable-image'));
        console.log(`Found ${allImages.length} clickable images`);

        // Add click handlers to all images
        allImages.forEach((img, index) => {
            img.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();
                openImageModal(index);
            });
        });

        // Create modal HTML
        createImageModal();

        // Keyboard navigation
        document.addEventListener('keydown', function(e) {
            if (currentImageModal && currentImageModal.classList.contains('show')) {
                e.preventDefault();
                if (e.key === 'Escape') {
                    closeImageModal();
                } else if (e.key === 'ArrowLeft') {
                    previousImage();
                } else if (e.key === 'ArrowRight') {
                    nextImage();
                }
            }
        });
    }

    function createImageModal() {
        const modal = document.createElement('div');
        modal.className = 'image-modal';
        modal.id = 'imageModal';

        modal.innerHTML = `
            <div class="modal-content">
                <button class="modal-close" onclick="closeImageModal()">×</button>
                <button class="modal-navigation modal-prev" onclick="previousImage()">‹</button>
                <button class="modal-navigation modal-next" onclick="nextImage()">›</button>

                <img class="modal-image" id="modalImage" alt="Modal Image">

                <div class="modal-info">
                    <div class="modal-title" id="modalTitle">Image Title</div>
                    <div class="modal-description" id="modalDescription">Image description</div>
                    <div class="modal-details" id="modalDetails"></div>
                </div>
            </div>
        `;

        document.body.appendChild(modal);
        currentImageModal = modal;

        // FIXED: Close modal when clicking outside - prevent event bubbling
        modal.addEventListener('click', function(e) {
            if (e.target === modal) {
                e.preventDefault();
                e.stopPropagation();
                closeImageModal();
            }
        });

        // FIXED: Prevent modal content clicks from closing modal
        const modalContent = modal.querySelector('.modal-content');
        modalContent.addEventListener('click', function(e) {
            e.stopPropagation();
        });
    }

    function openImageModal(index) {
        if (!allImages[index] || !currentImageModal) return;

         scrollPosition = window.pageYOffset || document.documentElement.scrollTop;
        currentImageIndex = index;
        const img = allImages[index];

        // Get image information
        const imageInfo = getImageInfo(img, index);

        // Update modal content
        document.getElementById('modalImage').src = img.src;
        document.getElementById('modalTitle').textContent = imageInfo.title;
        document.getElementById('modalDescription').textContent = imageInfo.description;

        // Update modal details
        const detailsContainer = document.getElementById('modalDetails');
        detailsContainer.innerHTML = '';

        imageInfo.details.forEach(detail => {
            const detailElement = document.createElement('div');
            detailElement.className = 'modal-detail';
            detailElement.textContent = detail;
            detailsContainer.appendChild(detailElement);
        });

        // FIXED: Proper modal display and body scroll handling
        currentImageModal.style.display = 'flex';
        requestAnimationFrame(() => {
            currentImageModal.classList.add('show');
            document.body.style.overflow = 'hidden';
            document.body.style.position = 'fixed';
            document.body.style.width = '100%';
        });

        // Update navigation buttons visibility
        const prevBtn = currentImageModal.querySelector('.modal-prev');
        const nextBtn = currentImageModal.querySelector('.modal-next');

        prevBtn.style.display = allImages.length > 1 ? 'flex' : 'none';
        nextBtn.style.display = allImages.length > 1 ? 'flex' : 'none';

        console.log(`Opened modal for image ${index + 1} of ${allImages.length}`);
    }

    function closeImageModal() {
        if (!currentImageModal || !currentImageModal.classList.contains('show')) return;

        console.log('Closing image modal...');

        // FIXED: Proper cleanup sequence
        currentImageModal.classList.remove('show');

        // FIXED: Restore body scroll immediately
        document.body.style.overflow = '';
        document.body.style.position = '';
        document.body.style.width = '';

        window.scrollTo(0, scrollPosition);
        // Hide modal after transition
        setTimeout(() => {
            if (currentImageModal) {
                currentImageModal.style.display = 'none';
            }
        }, 300);
        console.log('Image modal closed and scroll restored');
    }

    function previousImage() {
        if (allImages.length <= 1) return;

        currentImageIndex = (currentImageIndex - 1 + allImages.length) % allImages.length;
        updateModalContent(currentImageIndex);
    }

    function nextImage() {
        if (allImages.length <= 1) return;

        currentImageIndex = (currentImageIndex + 1) % allImages.length;
        updateModalContent(currentImageIndex);
    }

    // FIXED: Separate update function to avoid reopening modal
    function updateModalContent(index) {
        if (!allImages[index] || !currentImageModal) return;

        const img = allImages[index];
        const imageInfo = getImageInfo(img, index);

        // Update modal content without reopening
        document.getElementById('modalImage').src = img.src;
        document.getElementById('modalTitle').textContent = imageInfo.title;
        document.getElementById('modalDescription').textContent = imageInfo.description;

        const detailsContainer = document.getElementById('modalDetails');
        detailsContainer.innerHTML = '';

        imageInfo.details.forEach(detail => {
            const detailElement = document.createElement('div');
            detailElement.className = 'modal-detail';
            detailElement.textContent = detail;
            detailsContainer.appendChild(detailElement);
        });
    }

    function getImageInfo(img, index) {
        // Extract information from image and surrounding elements
        let title = img.alt || `Image ${index + 1}`;
        let description = 'AI-generated multimedia content';
        let details = [`Image ${index + 1} of ${allImages.length}`];

        // Try to get title from nearby elements
        const mediaCard = img.closest('.media-card');
        if (mediaCard) {
            const titleElement = mediaCard.querySelector('.media-title');
            const descElement = mediaCard.querySelector('.media-description');

            if (titleElement) title = titleElement.textContent.trim();
            if (descElement) description = descElement.textContent.trim();
        }

        // Determine image type and add relevant details
        const src = img.src.toLowerCase();
        const fileName = src.split('/').pop();

        if (src.includes('cover')) {
            details.push('Story Cover');
            details.push('Type: Title Image');
        } else if (src.includes('character')) {
            details.push('Character Portrait');
            details.push('Type: Character Design');
        } else if (src.includes('world')) {
            details.push('World Building');
            details.push('Type: Environment');
        } else if (src.includes('scene')) {
            const sceneMatch = fileName.match(/scene_(\d+)/);
            if (sceneMatch) {
                details.push(`Scene ${parseInt(sceneMatch[1]) + 1}`);
            }

            const perspectiveMatch = fileName.match(/perspective_(\d+)/);
            if (perspectiveMatch) {
                details.push(`Perspective ${parseInt(perspectiveMatch[1]) + 1}`);
            }

            details.push('Type: Scene Image');
        } else if (src.includes('transition')) {
            details.push('Scene Transition');
            details.push('Type: Transition Effect');
        } else if (src.includes('end')) {
            details.push('Story Conclusion');
            details.push('Type: Ending Image');
        }

        // Add technical details
        details.push(`File: ${fileName}`);

        return { title, description, details };
    }

    // FIXED: Audio player initialization
    function initializeAudioPlayer() {
        audioPlayer = document.getElementById('mainAudioPlayer');

        if (!audioPlayer) {
            console.log('No audio player found');
            return;
        }

        console.log('Audio player found, setting up...');

        // Audio event listeners
        audioPlayer.addEventListener('loadedmetadata', function() {
            console.log('Audio metadata loaded');
            updateAudioProgress();
        });

        audioPlayer.addEventListener('timeupdate', updateAudioProgress);

        audioPlayer.addEventListener('play', function() {
            isAudioPlaying = true;
            updatePlayPauseButton();
        });

        audioPlayer.addEventListener('pause', function() {
            isAudioPlaying = false;
            updatePlayPauseButton();
        });

        audioPlayer.addEventListener('ended', function() {
            isAudioPlaying = false;
            updatePlayPauseButton();
            document.getElementById('audioProgressDisplay').textContent = 'Audiobook completed';
        });
    }

    // FIXED: Show/hide audio player
    function showAudioPlayer() {
        const popup = document.getElementById('audioPlayerPopup');
        if (popup) {
            popup.classList.remove('hide');
            popup.classList.add('show');
            audioPlayerVisible = true;
            console.log('Audio player shown');
        }
    }

    function hideAudioPlayer() {
        const popup = document.getElementById('audioPlayerPopup');
        if (popup) {
            popup.classList.remove('show');
            popup.classList.add('hide');
            audioPlayerVisible = false;
            console.log('Audio player hidden');
        }
    }

    // FIXED: Toggle audio player visibility
    function toggleAudioPlayer() {
        console.log('Toggle audio player, currently visible:', audioPlayerVisible);

        if (audioPlayerVisible) {
            hideAudioPlayer();
            // set closeAudioBtn button text to show
            document.getElementById('closeAudioBtn').textContent = '<';
        } else {
            // set button text to hide
            document.getElementById('closeAudioBtn').textContent = '×';
            showAudioPlayer();
        }
    }

    // FIXED: Play/pause functionality
    function playPauseAudio() {
        if (!audioPlayer) {
            console.log('No audio player available');
            return;
        }

        if (isAudioPlaying) {
            audioPlayer.pause();
            console.log('Audio paused');
        } else {
            audioPlayer.play().then(() => {
                console.log('Audio playing');
            }).catch(error => {
                console.error('Error playing audio:', error);
            });
        }
    }

    // Update play/pause button text
    function updatePlayPauseButton() {
        const btn = document.getElementById('playPauseBtn');
        if (btn) {
            btn.textContent = isAudioPlaying ? '⏸️ Pause' : '▶️ Play';
        }
    }

    // Skip functions
    function skipBackward() {
        if (audioPlayer) {
            audioPlayer.currentTime = Math.max(0, audioPlayer.currentTime - 10);
            console.log('Skipped backward');
        }
    }

    function skipForward() {
        if (audioPlayer) {
            audioPlayer.currentTime = Math.min(audioPlayer.duration, audioPlayer.currentTime + 10);
            console.log('Skipped forward');
        }
    }

    // Update progress display
    function updateAudioProgress() {
        if (!audioPlayer) return;

        const current = audioPlayer.currentTime;
        const duration = audioPlayer.duration;

        if (duration && !isNaN(duration)) {
            const progress = (current / duration) * 100;
            const currentTime = formatTime(current);
            const totalTime = formatTime(duration);

            const display = document.getElementById('audioProgressDisplay');
            if (display) {
                display.textContent = `${currentTime} / ${totalTime} (${Math.round(progress)}%)`;
            }
        }
    }

    // Format time helper
    function formatTime(seconds) {
        if (!seconds || isNaN(seconds)) return '0:00';

        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = Math.floor(seconds % 60);
        return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
    }

    // Scroll animations
    function initializeScrollAnimations() {
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -100px 0px'
        };

        const observer = new IntersectionObserver(function(entries) {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('visible');
                }
            });
        }, observerOptions);

        // Observe all fade-in elements
        document.querySelectorAll('.fade-in-up').forEach(el => {
            observer.observe(el);
        });
    }

    // Smooth scrolling for any links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // FIXED: Error handling for media with proper cleanup
    document.querySelectorAll('img, video, audio').forEach(media => {
        media.addEventListener('error', function() {
            console.warn('Media failed to load:', this.src);
            if (this.tagName === 'IMG' && this.classList.contains('clickable-image')) {
                // Remove from clickable images array if it fails to load
                const index = allImages.indexOf(this);
                if (index > -1) {
                    allImages.splice(index, 1);
                }

                this.style.display = 'none';
                const placeholder = document.createElement('div');
                placeholder.style.cssText = 'height: 280px; background: var(--glass-bg); display: flex; align-items: center; justify-content: center; color: var(--secondary-color);';
                placeholder.textContent = 'Image not found';
                this.parentNode.replaceChild(placeholder, this);
            }
        });
    });

    // PDF Functions
    function downloadPDF(pdfPath, fileName) {
        console.log('Downloading PDF:', fileName);

        // Create download link
        const link = document.createElement('a');
        link.href = pdfPath;
        link.download = fileName;
        link.style.display = 'none';

        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);

        // Show success message
        showNotification(`📥 Downloading ${fileName}...`, 'success');
    }

    function openPDFFullscreen(pdfPath) {
        console.log('Opening PDF fullscreen:', pdfPath);

        // Open PDF in new window/tab
        const newWindow = window.open(pdfPath, '_blank');

        if (!newWindow) {
            showNotification('❌ Please allow popups to view PDF fullscreen', 'error');
        } else {
            showNotification('🔍 PDF opened in new tab', 'success');
        }
    }

    function printPDF(pdfPath) {
        console.log('Printing PDF:', pdfPath);

        // Open PDF for printing
        const printWindow = window.open(pdfPath, '_blank');

        if (printWindow) {
            printWindow.onload = function() {
                printWindow.print();
            };
            showNotification('🖨️ PDF opened for printing', 'success');
        } else {
            showNotification('❌ Please allow popups to print PDF', 'error');
        }
    }

    // Notification system
    function showNotification(message, type = 'info') {
        // Remove existing notification
        const existing = document.querySelector('.notification');
        if (existing) {
            existing.remove();
        }

        // Create notification
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;

        // Style notification
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: var(--surface-color);
            color: var(--primary-color);
            padding: 1rem 2rem;
            border-radius: 10px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.4);
            border: 1px solid var(--glass-border);
            z-index: 10000;
            animation: slideInDown 0.3s ease-out;
            max-width: 90%;
            text-align: center;
        `;

        document.body.appendChild(notification);

        // Remove after 3 seconds
        setTimeout(() => {
            notification.style.animation = 'slideOutUp 0.3s ease-in';
            setTimeout(() => {
                notification.remove();
            }, 300);
        }, 3000);
    }

    // PDF iframe error handling
    document.addEventListener('DOMContentLoaded', function() {
        // Handle PDF iframe errors
        document.querySelectorAll('.pdf-viewer').forEach(iframe => {
            iframe.addEventListener('error', function() {
                console.log('PDF iframe error, showing fallback');
                this.style.display = 'none';

                // Show fallback message
                const fallback = document.createElement('div');
                fallback.style.cssText = `
                    padding: 3rem;
                    text-align: center;
                    color: var(--secondary-color);
                    background: var(--glass-bg);
                `;
                fallback.innerHTML = `
                    <h3>PDF Preview Not Available</h3>
                    <p>Your browser doesn't support PDF embedding.</p>
                    <a href="${this.src.split('#')[0]}" download class="download-btn" style="margin-top: 1rem;">
                        📥 Download PDF Instead
                    </a>
                `;

                this.parentNode.insertBefore(fallback, this);
            });
        });
    });
    """
