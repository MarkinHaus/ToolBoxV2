#!/usr/bin/env node

// ToolBox Pro - Build System
// Production build script for browser extension

const fs = require('fs').promises;
const path = require('path');
const { execSync } = require('child_process');

class ExtensionBuilder {
    constructor() {
        this.srcDir = path.join(__dirname, 'src');
        this.buildDir = path.join(__dirname, 'build');
        this.assetsDir = path.join(__dirname, 'assets');
        this.iconsDir = path.join(__dirname, 'icons');

        this.filesToCopy = [
            'manifest.json',
            'icons/tb16.png',
            'icons/tb32.png',
            'icons/tb48.png',
            'icons/tb128.png'
        ];
    }

    async build() {
        console.log('🚀 Building ToolBox Pro Extension...');

        try {
            // Clean build directory
            await this.cleanBuildDir();

            // Create build directory structure
            await this.createBuildStructure();

            // Copy and process files
            await this.copyFiles();

            // Generate icons
            //await this.generateIcons();

            // Minify files for production
            await this.minifyFiles();

            // Validate build
            await this.validateBuild();

            console.log('✅ Build completed successfully!');
            console.log(`📦 Extension ready in: ${this.buildDir}`);

        } catch (error) {
            console.error('❌ Build failed:', error);
            process.exit(1);
        }
    }

    async cleanBuildDir() {
        console.log('🧹 Cleaning build directory...');

        try {
            await fs.rmdir(this.buildDir, { recursive: true });
        } catch (error) {
            // Directory might not exist, that's okay
        }
    }

    async createBuildStructure() {
        console.log('📁 Creating build structure...');

        const dirs = [
            this.buildDir,
            path.join(this.buildDir, 'src'),
            path.join(this.buildDir, 'icons'),
            path.join(this.buildDir, 'assets')
        ];

        for (const dir of dirs) {
            await fs.mkdir(dir, { recursive: true });
        }
    }

    async copyRecursive(src, dest) {
        await fs.mkdir(dest, { recursive: true });
        const entries = await fs.readdir(src, { withFileTypes: true });
        for (const entry of entries) {
            const srcPath = path.join(src, entry.name);
            const destPath = path.join(dest, entry.name);
            if (entry.isDirectory()) {
                await this.copyRecursive(srcPath, destPath);
            } else {
                await fs.copyFile(srcPath, destPath);
                console.log(`  ✓ Copied: ${path.relative(__dirname, srcPath)}`);
            }
        }
    }

    async copyFiles() {
        console.log('📄 Copying files...');

        // 1. Kopiere die Dateien aus der Liste (manifest + icons)
        for (const file of this.filesToCopy) {
            const srcPath = path.join(__dirname, file);
            const destPath = path.join(this.buildDir, file);

            try {
                // Erstelle Unterverzeichnisse falls nötig (z.B. für icons/)
                await fs.mkdir(path.dirname(destPath), { recursive: true });
                await fs.copyFile(srcPath, destPath);
                console.log(`  ✓ ${file}`);
            } catch (error) {
                console.warn(`  ⚠️ Failed to copy ${file}:`, error.message);
            }
        }

        // 2. Kopiere den gesamten src-Ordner REKURSIV
        console.log('📂 Copying src folder...');
        await fs.cp(this.srcDir, path.join(this.buildDir, 'src'), { recursive: true });
        console.log('  ✓ src directory copied');
    }

    processJavaScript(content) {
        // Remove console.log statements in production
        return content.replace(/console\.(log|info|debug)\([^)]*\);?\s*/g, '');
    }

    processCSS(content) {
        // Remove comments and extra whitespace
        return content
            .replace(/\/\*[\s\S]*?\*\//g, '')
            .replace(/\s+/g, ' ')
            .trim();
    }

    processHTML(content) {
        // Remove comments and extra whitespace
        return content
            .replace(/<!--[\s\S]*?-->/g, '')
            .replace(/\s+/g, ' ')
            .replace(/>\s+</g, '><')
            .trim();
    }

    processJSON(content) {
        // Minify JSON
        const parsed = JSON.parse(content);
        return JSON.stringify(parsed, null, 0);
    }

    async generateIcons() {
        console.log('🎨 Generating icons...');

        // Create simple SVG icon
        const iconSVG = `
            <svg width="128" height="128" viewBox="0 0 128 128" xmlns="http://www.w3.org/2000/svg">
                <defs>
                    <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" style="stop-color:#2E86AB;stop-opacity:1" />
                        <stop offset="100%" style="stop-color:#A23B72;stop-opacity:1" />
                    </linearGradient>
                </defs>
                <rect width="128" height="128" rx="24" fill="url(#grad)"/>
                <text x="64" y="80" font-family="Arial, sans-serif" font-size="48" font-weight="bold" text-anchor="middle" fill="white">TB</text>
                <circle cx="96" cy="32" r="8" fill="white" opacity="0.8"/>
            </svg>
        `;

        // Save SVG
        await fs.writeFile(path.join(this.buildDir, 'icons', 'tb.svg'), iconSVG);

        // Generate different sizes (placeholder - in real implementation, you'd use a proper image processing library)
        const sizes = [16, 32, 48, 128];
        for (const size of sizes) {
            const placeholder = `data:image/svg+xml;base64,${Buffer.from(iconSVG).toString('base64')}`;
            await fs.writeFile(
                path.join(this.buildDir, 'icons', `tb${size}.png`),
                `<!-- Placeholder for ${size}x${size} PNG icon -->\n<!-- In production, convert SVG to PNG -->`
            );
        }

        console.log('  ✓ Generated icon files');
    }

    async minifyFiles() {
        console.log('🗜️ Minifying files...');

        // In a real implementation, you would use proper minification tools
        // For now, we'll do basic minification

        const jsFiles = [
            'src/popup.js',
            'src/content.js',
            'src/background.js',
            'src/content/injector.js'
        ];

        for (const file of jsFiles) {
            const filePath = path.join(this.buildDir, file);
            try {
                let content = await fs.readFile(filePath, 'utf8');

                // Basic minification
                content = content
                    .replace(/\/\*[\s\S]*?\*\//g, '') // Remove block comments
                    .replace(/\/\/.*$/gm, '') // Remove line comments
                    .replace(/\s+/g, ' ') // Collapse whitespace
                    .replace(/;\s*}/g, '}') // Remove semicolons before closing braces
                    .trim();

                await fs.writeFile(filePath, content);
                console.log(`  ✓ Minified ${file}`);
            } catch (error) {
                console.warn(`  ⚠️ Failed to minify ${file}:`, error.message);
            }
        }
    }

    async validateBuild() {
        console.log('🔍 Validating build...');

        // Check if all required files exist
        const requiredFiles = [
            'manifest.json',
            'src/popup.html',
            'src/popup.js',
            'src/site_rules.json',
            'src/library.json',
            'src/popup.css',
            'src/content.js',
            'src/agent_view.js',
            'src/prompt_engine.js',
            'src/injector.js',
            'src/content.css',
            'src/gesture-detector.js',
            'src/background.js',
            'icons/tb16.png',
            'icons/tb32.png',
            'icons/tb48.png',
            'icons/tb128.png'
        ];

        let allFilesExist = true;

        for (const file of requiredFiles) {
            const filePath = path.join(this.buildDir, file);
            try {
                await fs.access(filePath);
                console.log(`  ✓ ${file}`);
            } catch (error) {
                console.error(`  ❌ Missing: ${file}`);
                allFilesExist = false;
            }
        }

        if (!allFilesExist) {
            throw new Error('Build validation failed - missing required files');
        }

        // Validate manifest.json
        try {
            const manifestPath = path.join(this.buildDir, 'manifest.json');
            const manifestContent = await fs.readFile(manifestPath, 'utf8');
            const manifest = JSON.parse(manifestContent);

            if (!manifest.manifest_version || !manifest.name || !manifest.version) {
                throw new Error('Invalid manifest.json');
            }

            console.log(`  ✓ Manifest valid (v${manifest.version})`);
        } catch (error) {
            throw new Error(`Manifest validation failed: ${error.message}`);
        }
    }

    async createZip() {
        console.log('📦 Creating distribution package...');

        const zipName = `toolbox-pro-extension-v${await this.getVersion()}.zip`;
        const zipPath = path.join(__dirname, zipName);

        try {
            // Use system zip command if available
            execSync(`cd "${this.buildDir}" && zip -r "${zipPath}" .`, { stdio: 'inherit' });
            console.log(`✅ Package created: ${zipName}`);
        } catch (error) {
            console.warn('⚠️ Could not create zip package. Please zip the build folder manually.');
        }
    }

    async getVersion() {
        try {
            const manifestPath = path.join(__dirname, 'manifest.json');
            const manifest = JSON.parse(await fs.readFile(manifestPath, 'utf8'));
            return manifest.version;
        } catch (error) {
            return '3.0.0';
        }
    }

    async dev() {
        console.log('🔧 Starting development build...');

        // Clean and create structure
        await this.cleanBuildDir();
        await this.createBuildStructure();

        // Copy files without minification
        for (const file of this.filesToCopy) {
            const srcPath = path.join(__dirname, file);
            const destPath = path.join(this.buildDir, file);

            try {
                await fs.copyFile(srcPath, destPath);
                console.log(`  ✓ ${file}`);
            } catch (error) {
                console.warn(`  ⚠️ Failed to copy ${file}:`, error.message);
            }
        }

        // Generate icons
        //await this.generateIcons();

        console.log('✅ Development build ready!');
        console.log('💡 Load the extension from the build folder in Chrome Developer Mode');
    }
}

// CLI interface
const builder = new ExtensionBuilder();

const command = process.argv[2];

switch (command) {
    case 'dev':
        builder.dev();
        break;
    case 'build':
        builder.build();
        break;
    case 'zip':
        builder.build().then(() => builder.createZip());
        break;
    default:
        console.log('Usage: node build.js [dev|build|zip]');
        console.log('  dev   - Development build (no minification)');
        console.log('  build - Production build');
        console.log('  zip   - Production build + create zip package');
        process.exit(1);
}
