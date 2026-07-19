/**
 * Password strength checker with real-time validation
 */
function initPasswordValidator() {
    const passwordInput = document.getElementById('password');
    const confirmPasswordInput = document.getElementById('confirm_password');
    
    if (!passwordInput) return; // Exit if we're not on a page with password fields
    
    // Create password strength meter elements
    const strengthMeter = document.createElement('div');
    strengthMeter.className = 'password-strength-meter';
    
    const strengthFill = document.createElement('div');
    strengthFill.className = 'password-strength-meter-fill';
    
    const strengthText = document.createElement('div');
    strengthText.className = 'password-strength-text';
    
    strengthMeter.appendChild(strengthFill);
    
    // Insert elements after password input
    passwordInput.parentNode.insertBefore(strengthMeter, passwordInput.nextSibling);
    passwordInput.parentNode.insertBefore(strengthText, strengthMeter.nextSibling);
    
    // Create validation message element for matching
    const matchMessage = document.createElement('div');
    matchMessage.className = 'password-match-message';
    matchMessage.style.fontSize = '0.8rem';
    matchMessage.style.marginTop = '0.25rem';
    
    if (confirmPasswordInput) {
        confirmPasswordInput.parentNode.insertBefore(matchMessage, confirmPasswordInput.nextSibling);
    }
    
    // Password criteria
    const criteria = [
        { regex: /.{8,}/, description: "At least 8 characters" },
        { regex: /[A-Z]/, description: "At least one uppercase letter" },
        { regex: /[a-z]/, description: "At least one lowercase letter" },
        { regex: /[0-9]/, description: "At least one number" },
        { regex: /[^A-Za-z0-9]/, description: "At least one special character" }
    ];
    
    // Create criteria list
    const criteriaList = document.createElement('ul');
    criteriaList.className = 'password-criteria-list';
    criteriaList.style.fontSize = '0.8rem';
    criteriaList.style.marginTop = '0.5rem';
    criteriaList.style.paddingLeft = '1.2rem';
    
    criteria.forEach(criterion => {
        const item = document.createElement('li');
        item.textContent = criterion.description;
        item.style.color = 'rgba(255, 255, 255, 0.6)';
        item.style.transition = 'color 0.3s ease';
        item.dataset.description = criterion.description;
        criteriaList.appendChild(item);
    });
    
    // Add criteria list after strength text
    passwordInput.parentNode.insertBefore(criteriaList, strengthText.nextSibling);
    
    // Check password strength when typing
    passwordInput.addEventListener('input', function() {
        const password = this.value;
        let strength = 0;
        
        // Update criteria list
        criteriaList.querySelectorAll('li').forEach((item, index) => {
            const isMet = criteria[index].regex.test(password);
            item.style.color = isMet ? '#2ecc71' : 'rgba(255, 255, 255, 0.6)';
            
            if (isMet) {
                strength += 1;
            }
        });
        
        // Calculate percentage
        const strengthPercent = (strength / criteria.length) * 100;
        
        // Update meter fill
        strengthFill.style.width = `${strengthPercent}%`;
        
        // Set color and text based on strength
        if (strengthPercent === 0) {
            strengthFill.style.backgroundColor = 'transparent';
            strengthText.textContent = '';
        } else if (strengthPercent <= 30) {
            strengthFill.style.backgroundColor = '#e74c3c'; // Red
            strengthText.textContent = 'Weak';
            strengthText.style.color = '#e74c3c';
        } else if (strengthPercent <= 60) {
            strengthFill.style.backgroundColor = '#f39c12'; // Orange
            strengthText.textContent = 'Moderate';
            strengthText.style.color = '#f39c12';
        } else if (strengthPercent <= 80) {
            strengthFill.style.backgroundColor = '#3498db'; // Blue
            strengthText.textContent = 'Good';
            strengthText.style.color = '#3498db';
        } else {
            strengthFill.style.backgroundColor = '#2ecc71'; // Green
            strengthText.textContent = 'Strong';
            strengthText.style.color = '#2ecc71';
        }
        
        // If there's a confirm password field, check matching
        if (confirmPasswordInput && confirmPasswordInput.value) {
            checkPasswordsMatch();
        }
    });
    
    // Check if passwords match
    function checkPasswordsMatch() {
        if (!confirmPasswordInput) return;
        
        const password = passwordInput.value;
        const confirmPassword = confirmPasswordInput.value;
        
        if (confirmPassword) {
            if (password === confirmPassword) {
                matchMessage.textContent = 'Passwords match';
                matchMessage.style.color = '#2ecc71'; // Green
            } else {
                matchMessage.textContent = 'Passwords do not match';
                matchMessage.style.color = '#e74c3c'; // Red
            }
        } else {
            matchMessage.textContent = '';
        }
    }
    
    // Add event listener to confirm password
    if (confirmPasswordInput) {
        confirmPasswordInput.addEventListener('input', checkPasswordsMatch);
    }
}

// Initialize animations
function initAnimations() {
    // Add fade-in animation to buttons
    const buttons = document.querySelectorAll('button');
    buttons.forEach((button, index) => {
        button.style.animation = `fadeIn 0.5s ease ${index * 0.1}s both`;
    });
    
    // Add animation to form inputs on focus
    const inputs = document.querySelectorAll('input, textarea, select');
    inputs.forEach(input => {
        input.addEventListener('focus', () => {
            input.style.transform = 'translateY(-2px)';
            input.style.boxShadow = '0 4px 8px rgba(0,0,0,0.1)';
        });
        
        input.addEventListener('blur', () => {
            input.style.transform = 'translateY(0)';
            input.style.boxShadow = 'none';
        });
    });
}

// Initialize everything when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initPasswordValidator();
    initAnimations();
    
    // Form submissions with animations
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        const submitBtn = form.querySelector('button[type="submit"]');
        if (submitBtn) {
            const originalText = submitBtn.textContent;
            
            form.addEventListener('submit', function() {
                // Add loading animation if not handled by SweetAlert
                if (!window.Swal) {
                    submitBtn.innerHTML = '<span class="spinner"></span> Processing...';
                    submitBtn.disabled = true;
                }
            });
        }
    });
});