// Image manipulation functions
class ImageEditor {
    constructor(imageElement) {
        this.image = imageElement;
        this.canvas = document.createElement('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.originalImage = new Image();
        this.originalImage.src = imageElement.src;
        this.currentImage = new Image();
        this.currentImage.src = imageElement.src;
        this.init();
    }

    init() {
        this.canvas.width = this.image.width;
        this.canvas.height = this.image.height;
        this.ctx.drawImage(this.image, 0, 0);
    }


    rotate(degrees) {
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        
        // Calculate new canvas size
        const radians = degrees * Math.PI / 180;
        const sin = Math.abs(Math.sin(radians));
        const cos = Math.abs(Math.cos(radians));
        const newWidth = this.currentImage.width * cos + this.currentImage.height * sin;
        const newHeight = this.currentImage.width * sin + this.currentImage.height * cos;
        
        tempCanvas.width = newWidth;
        tempCanvas.height = newHeight;
        
        // Rotate around center
        tempCtx.translate(newWidth / 2, newHeight / 2);
        tempCtx.rotate(radians);
        tempCtx.drawImage(this.currentImage, -this.currentImage.width / 2, -this.currentImage.height / 2);
        
        this.currentImage.src = tempCanvas.toDataURL();
        this.updateImage();
    }

    flip(axis) {
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        tempCanvas.width = this.currentImage.width;
        tempCanvas.height = this.currentImage.height;
        
        if (axis === 'horizontal') {
            tempCtx.translate(this.currentImage.width, 0);
            tempCtx.scale(-1, 1);
        } else if (axis === 'vertical') {
            tempCtx.translate(0, this.currentImage.height);
            tempCtx.scale(1, -1);
        }
        
        tempCtx.drawImage(this.currentImage, 0, 0);
        this.currentImage.src = tempCanvas.toDataURL();
        this.updateImage();
    }

    applyFilter(filter) {
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        tempCanvas.width = this.currentImage.width;
        tempCanvas.height = this.currentImage.height;
        tempCtx.drawImage(this.currentImage, 0, 0);
        
        const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
        const data = imageData.data;
        
        switch(filter) {
            case 'grayscale':
                for (let i = 0; i < data.length; i += 4) {
                    const avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
                    data[i] = avg;
                    data[i + 1] = avg;
                    data[i + 2] = avg;
                }
                break;
            case 'sepia':
                for (let i = 0; i < data.length; i += 4) {
                    const r = data[i];
                    const g = data[i + 1];
                    const b = data[i + 2];
                    data[i] = Math.min(255, (r * 0.393) + (g * 0.769) + (b * 0.189));
                    data[i + 1] = Math.min(255, (r * 0.349) + (g * 0.686) + (b * 0.168));
                    data[i + 2] = Math.min(255, (r * 0.272) + (g * 0.534) + (b * 0.131));
                }
                break;
            case 'invert':
                for (let i = 0; i < data.length; i += 4) {
                    data[i] = 255 - data[i];
                    data[i + 1] = 255 - data[i + 1];
                    data[i + 2] = 255 - data[i + 2];
                }
                break;
        }
        
        tempCtx.putImageData(imageData, 0, 0);
        this.currentImage.src = tempCanvas.toDataURL();
        this.updateImage();
    }

    adjustBrightness(value) {
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        tempCanvas.width = this.currentImage.width;
        tempCanvas.height = this.currentImage.height;
        tempCtx.drawImage(this.currentImage, 0, 0);
        
        const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
        const data = imageData.data;
        
        for (let i = 0; i < data.length; i += 4) {
            data[i] = Math.min(255, Math.max(0, data[i] + value));
            data[i + 1] = Math.min(255, Math.max(0, data[i + 1] + value));
            data[i + 2] = Math.min(255, Math.max(0, data[i + 2] + value));
        }
        
        tempCtx.putImageData(imageData, 0, 0);
        this.currentImage.src = tempCanvas.toDataURL();
        this.updateImage();
    }

    updateImage() {
        this.image.src = this.currentImage.src;
    }

    reset() {
        this.currentImage.src = this.originalImage.src;
        this.updateImage();
    }
}

document.addEventListener('DOMContentLoaded', function() {
    // Initialize image comparison slider
    const initImageComparison = () => {
        const comparisonWrapper = document.querySelector('.comparison-wrapper');
        const slider = document.querySelector('.comparison-slider');
        const sliderHandle = document.querySelector('.slider-handle');
        const originalImage = document.querySelector('.comparison-image.original');
        const enhancedImage = document.querySelector('.comparison-image.enhanced');
        
        if (!comparisonWrapper || !slider || !sliderHandle) return;

        let isDragging = false;

        const updateComparison = (clientX) => {
            const rect = comparisonWrapper.getBoundingClientRect();
            let position = clientX - rect.left;
            position = Math.max(0, Math.min(position, rect.width));
            
            const percent = (position / rect.width) * 100;
            slider.style.left = `${percent}%`;
            originalImage.style.clipPath = `inset(0 ${100 - percent}% 0 0)`;
            enhancedImage.style.clipPath = `inset(0 0 0 ${percent}%)`;
        };

        const handleMove = (e) => {
            if (!isDragging) return;
            const clientX = e.touches ? e.touches[0].clientX : e.clientX;
            updateComparison(clientX);
        };

        const handleEnd = () => {
            isDragging = false;
            document.removeEventListener('mousemove', handleMove);
            document.removeEventListener('touchmove', handleMove);
            document.removeEventListener('mouseup', handleEnd);
            document.removeEventListener('touchend', handleEnd);
        };

        sliderHandle.addEventListener('mousedown', (e) => {
            isDragging = true;
            document.addEventListener('mousemove', handleMove);
            document.addEventListener('mouseup', handleEnd);
        });

        sliderHandle.addEventListener('touchstart', (e) => {
            isDragging = true;
            document.addEventListener('touchmove', handleMove);
            document.addEventListener('touchend', handleEnd);
        });

        comparisonWrapper.addEventListener('click', (e) => {
            const clientX = e.touches ? e.touches[0].clientX : e.clientX;
            updateComparison(clientX);
        });

        // Initialize slider position
        updateComparison(comparisonWrapper.getBoundingClientRect().width / 2);
    };

    // Initialize image editor if on results page
    const enhancedImage = document.querySelector('.result-image');
    if (enhancedImage) {
        const editor = new ImageEditor(enhancedImage);
        
        // Add event listeners for editor controls
        document.querySelectorAll('.editor-control').forEach(control => {
            control.addEventListener('click', function() {
                const action = this.dataset.action;
                const value = this.dataset.value;
                
                switch(action) {
                    case 'rotate':
                        editor.rotate(parseInt(value));
                        break;
                    case 'flip':
                        editor.flip(value);
                        break;
                    case 'filter':
                        editor.applyFilter(value);
                        break;
                    case 'brightness':
                        editor.adjustBrightness(parseInt(value));
                        break;
                    case 'reset':
                        editor.reset();
                        break;
                }
            });
        });
    }

    // Initialize image comparison slider
    initImageComparison();

    // Page transition effect
    const pageTransition = document.querySelector('.page-transition');
    
    if (pageTransition) {
        // Show transition when page loads
        pageTransition.classList.add('active');
        
        // Hide transition after a delay
        setTimeout(() => {
            pageTransition.classList.remove('active');
        }, 500);
    }
    
    // Add animation classes to elements
    const fadeElements = document.querySelectorAll('.fade-in');
    const slideElements = document.querySelectorAll('.slide-up');
    
    if (fadeElements.length > 0 || slideElements.length > 0) {
        // Create an intersection observer for animations
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.style.opacity = 1;
                    entry.target.style.transform = 'translateY(0)';
                    observer.unobserve(entry.target);
                }
            });
        }, {
            threshold: 0.1,
            rootMargin: '0px 0px 50px 0px'
        });
        
        // Observe fade elements
        fadeElements.forEach(element => {
            element.style.opacity = 0;
            observer.observe(element);
        });
        
        // Observe slide elements
        slideElements.forEach(element => {
            element.style.opacity = 0;
            element.style.transform = 'translateY(50px)';
            observer.observe(element);
        });
    }
    
    // Mobile menu toggle
    const menuToggle = document.querySelector('.menu-toggle');
    const navLinks = document.querySelector('.nav-links');
    
    if (menuToggle && navLinks) {
        menuToggle.addEventListener('click', () => {
            navLinks.classList.toggle('active');
            menuToggle.classList.toggle('active');
        });
    }
    
    // Close flash messages
    const closeButtons = document.querySelectorAll('.close-flash');
    
    closeButtons.forEach(button => {
        button.addEventListener('click', () => {
            const flashMessage = button.parentElement;
            flashMessage.style.opacity = 0;
            setTimeout(() => {
                flashMessage.remove();
            }, 300);
        });
    });
    
    // Auto-hide flash messages after 5 seconds
    const flashMessages = document.querySelectorAll('.flash-message');
    
    flashMessages.forEach(message => {
        setTimeout(() => {
            message.style.opacity = 0;
            setTimeout(() => {
                message.remove();
            }, 300);
        }, 5000);
    });
    
    // File upload functionality
    const uploadArea = document.querySelector('.upload-area');
    const fileInput = document.querySelector('#file-input');
    
    if (uploadArea && fileInput) {
        uploadArea.addEventListener('click', () => {
            fileInput.click();
        });
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('active');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('active');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('active');
            
            if (e.dataTransfer.files.length) {
                fileInput.files = e.dataTransfer.files;
                const form = fileInput.closest('form');
                if (form) form.submit();
            }
        });
        
        fileInput.addEventListener('change', () => {
            if (fileInput.files.length) {
                const form = fileInput.closest('form');
                if (form) form.submit();
            }
        });
    }
    
    // Scroll to top button
    const scrollTopButton = document.querySelector('.scroll-top');
    
    if (scrollTopButton) {
        const scrollHandler = () => {
            if (window.pageYOffset > 300) {
                scrollTopButton.classList.add('show');
            } else {
                scrollTopButton.classList.remove('show');
            }
        };
        
        // Use passive event listener for better scroll performance
        window.addEventListener('scroll', scrollHandler, { passive: true });
        
        scrollTopButton.addEventListener('click', () => {
            window.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
        });
    }
    
    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            const targetId = this.getAttribute('href');
            
            if (targetId === '#') return;
            
            e.preventDefault();
            
            const targetElement = document.querySelector(targetId);
            
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
    });
}); 