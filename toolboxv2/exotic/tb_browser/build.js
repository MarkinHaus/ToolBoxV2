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
            await this.cleanBuildDir();
            await this.createBuildStructure();
            await this.copyFiles();
            await this.minifyFiles();
            await this.validateBuild();

            console.log('✅ Build completed successfully!');
            console.log(`📦 Extension ready in: ${this.buildDir}`);

        } catch (error) {
            console.error('❌ Build failed:', error);
            process.exit(1);
        }
    }

    // FIX 1: fs.rmdir({ recursive }) is deprecated since Node 16 → use fs.rm
    async cleanBuildDir() {
        console.log('🧹 Cleaning build directory...');
        await fs.rm(this.buildDir, { recursive: true, force: true });
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

        for (const file of this.filesToCopy) {
            const srcPath = path.join(__dirname, file);
            const destPath = path.join(this.buildDir, file);

            try {
                await fs.mkdir(path.dirname(destPath), { recursive: true });
                await fs.copyFile(srcPath, destPath);
                console.log(`  ✓ ${file}`);
            } catch (error) {
                console.warn(`  ⚠️ Failed to copy ${file}:`, error.message);
            }
        }

        // Copy entire src folder recursively
        console.log('📂 Copying src folder...');
        await fs.cp(this.srcDir, path.join(this.buildDir, 'src'), { recursive: true });
        console.log('  ✓ src directory copied');
    }

    processJavaScript(content) {
        return content.replace(/console\.(log|info|debug)\([^)]*\);?\s*/g, '');
    }

    processCSS(content) {
        return content
            .replace(/\/\*[\s\S]*?\*\//g, '')
            .replace(/\s+/g, ' ')
            .trim();
    }

    processHTML(content) {
        return content
            .replace(/<!--[\s\S]*?-->/g, '')
            .replace(/\s+/g, ' ')
            .replace(/>\s+</g, '><')
            .trim();
    }

    processJSON(content) {
        const parsed = JSON.parse(content);
        return JSON.stringify(parsed, null, 0);
    }

    async minifyFiles() {
        console.log('🗜️ Minifying files...');

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

                content = content
                    .replace(/\/\*[\s\S]*?\*\//g, '')
                    .replace(/\/\/.*$/gm, '')
                    .replace(/\s+/g, ' ')
                    .replace(/;\s*}/g, '}')
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

        // FIX 3: Paths corrected to match actual folder structure
        //   src/prompts/site_rules.json  (not src/site_rules.json)
        //   src/prompts/library.json     (not src/library.json)
        //   src/prompts/prompt_engine.js (not src/prompt_engine.js)
        //   src/content/injector.js      (not src/injector.js)
        const requiredFiles = [
            'manifest.json',
            'src/popup.html',
            'src/popup.js',
            'src/popup.css',
            'src/content.js',
            'src/content.css',
            'src/agent_view.js',
            'src/background.js',
            'src/gesture-detector.js',
            'src/prompts/site_rules.json',
            'src/prompts/library.json',
            'src/prompts/prompt_engine.js',
            'src/content/injector.js',
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

    // FIX 2: dev() now calls copyFiles() so src/ is actually copied
    async dev() {
        console.log('🔧 Starting development build...');

        await this.cleanBuildDir();
        await this.createBuildStructure();
        await this.copyFiles();

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
