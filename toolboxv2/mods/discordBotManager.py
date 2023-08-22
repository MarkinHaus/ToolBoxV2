import asyncio
import os
import queue
import tempfile
import threading
import time
from urllib.parse import quote_plus

import aiohttp
import discord
import requests
from discord.ext import commands
from toolboxv2 import MainTool, FileHandler, App, Style, remove_styles
from toolboxv2.mods.isaa import Tools as isaaTools
from toolboxv2.mods.cloudM import test_if_exists, Tools as cmTools


class Dropdown(discord.ui.Select):
    def __init__(self, options, callback_func=None, placeholder="Select on"):
        # Set the options that will be presented inside the dropdown

        # options = [
        #    discord.SelectOption(label='Red', description='Your favourite colour is red', emoji='🟥'),
        #    discord.SelectOption(label='Green', description='Your favourite colour is green', emoji='🟩'),
        #    discord.SelectOption(label='Blue', description='Your favourite colour is blue', emoji='🟦'),
        # ]

        # The placeholder is what will be shown when no option is chosen
        # The min and max values indicate we can only pick one of the three options
        # The options parameter defines the dropdown options. We defined this above
        super().__init__(placeholder=placeholder, min_values=1, max_values=1, options=options)
        self.callback_func = callback_func

    async def callback(self, interaction: discord.Interaction):
        # Use the interaction object to send a response message containing
        # the user's favourite colour or choice. The self object refers to the
        # Select object, and the values attribute gets a list of the user's
        # selected options. We only want the first one.
        msg = f'Selected : {self.values[0]}'
        if self.callback_func is not None:
            await self.callback_func(self.values[0], interaction.response.send_message)
        else:
            await interaction.response.send_message(msg)


class DropdownView(discord.ui.View):
    def __init__(self, options):
        super().__init__()

        # Adds the dropdown to our view object.
        self.add_item(Dropdown(options))


class Confirm(discord.ui.View):
    def __init__(self):
        super().__init__()
        self.value = None

    # When the confirm button is pressed, set the inner value to `True` and
    # stop the View from listening to more input.
    # We also send the user an ephemeral message that we're confirming their choice.
    @discord.ui.button(label='Confirm', style=discord.ButtonStyle.green)
    async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_message('Confirming', ephemeral=True)
        self.value = True
        self.stop()

    # This one is similar to the confirmation button except sets the inner value to `False`
    @discord.ui.button(label='Cancel', style=discord.ButtonStyle.grey)
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_message('Cancelling', ephemeral=True)
        self.value = False
        self.stop()


# Define a View that will give us our own personal counter button
class EphemeralCounter(discord.ui.View):
    def __init__(self, view, msg):
        self.view = view
        self.msg = msg

    # When this button is pressed, it will respond with a Counter view that will
    # give the button presser their own personal button they can press 5 times.
    @discord.ui.button(label='Click', style=discord.ButtonStyle.blurple)
    async def receive(self, interaction: discord.Interaction, button: discord.ui.Button):
        # ephemeral=True makes the message hidden from everyone except the button presser
        await interaction.response.send_message(self.msg, view=self.view, ephemeral=True)


class PersistentView(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=None)

    @discord.ui.button(label='Green', style=discord.ButtonStyle.green, custom_id='persistent_view:green')
    async def green(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_message('This is green.', ephemeral=True)

    @discord.ui.button(label='Red', style=discord.ButtonStyle.red, custom_id='persistent_view:red')
    async def red(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_message('This is red.', ephemeral=True)

    @discord.ui.button(label='Grey', style=discord.ButtonStyle.grey, custom_id='persistent_view:grey')
    async def grey(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_message('This is grey.', ephemeral=True)


class TaskSelectionView(discord.ui.View):
    def __init__(self, run_chain_callback, options, selected, chains):
        super().__init__(timeout=60 * 15)
        if len(options) > 23:
            options = options[:23]
        self.options = options
        self.run_chain_callback = run_chain_callback
        self.selected = selected
        options_select = [
            discord.SelectOption(value="1", label='Validate', description=f'run {selected} chain on task', emoji='🟩'),
            discord.SelectOption(value="-1", label='Abort', description='Cancel', emoji='🟥'),
        ]

        for _ in options:
            dis = chains.get_discr(_)
            if dis is None:
                dis = _
            if len(dis) > 100:
                dis = dis[:96]+'...'
            options_select.append(discord.SelectOption(value=_, label=_, description=dis))

        async def callback_fuc(val, send):
            if val == "1":
                await send("Validate ...")
                await run_chain_callback(selected)
            elif val == "-1":
                return "Abort"
            else:
                if val in options:
                    await send(f"Validate {val}...")
                    await run_chain_callback(val)
                else:
                    await send(f"invalid chain name Valid ar : {options}")
            return "Don"

        self.add_item(Dropdown(options_select, callback_fuc))


class Google(discord.ui.View):
    def __init__(self, query: str):
        super().__init__()
        # we need to quote the query string to make a valid url. Discord will raise an error if it isn't valid.
        query = quote_plus(query)
        url = f'https://www.google.com/search?q={query}'

        # Link buttons cannot be made with the decorator
        # Therefore we have to manually create one.
        # We add the quoted url to the button, and add the button to the view.
        self.add_item(discord.ui.Button(label='Click Here', url=url))


class Tools(commands.Bot, MainTool):
    def __init__(self, app=None, command_prefix=''):
        intents = discord.Intents.default()
        intents.message_content = True
        commands.Bot.__init__(self, command_prefix=command_prefix, intents=intents, self_bot=False)
        self.add_commands()

        self.guild = None
        self.version = '0.0.1'
        self.name = 'discordBotManager'
        self.logger = app.logger if app else None
        self.color = 'WHITE'
        self.token = ""
        self.context = []
        self.t0 = None
        self.sender_que = None
        self.receiver_que = None
        self.tools = {
            'all': [['Version', 'Shows current Version'],
                    ['start_bot', 'Start then Discord Bot'],
                    ['stop_bot', 'Stop the Discord Bot'],
                    ],
            'name': 'discordBotManager',
            'Version': self.show_version,
            'stop_bot': self.stop_bot,
            'start_bot': self.start_bot,
        }
        self.voice_index = 0
        self.ws_id = ""
        MainTool.__init__(self, load=self.on_start, v=self.version, tool=self.tools, name=self.name, logs=self.logger,
                          color=self.color, on_exit=self.on_exit)
        self.isaa: isaaTools = app.get_mod('isaa')
        self.most_resent_msg = None

    async def on_ready(self):
        guild = discord.utils.get(self.guilds, name='ISAA')
        if not guild:
            guild = await self.create_guild(name='ISAA')
        self.guild = guild
        for channel_name in ['context', 'options', 'chat']:
            if not discord.utils.get(guild.text_channels, name=channel_name):
                await guild.create_text_channel(channel_name)

        if not discord.utils.get(guild.voice_channels, name="speak-with-isaa"):
            await guild.create_text_channel("speak-with-isaa")

    async def minique_responder(self, ctx):
        running = True

        async def send(x):
            t = 60 * ((len(x) // 100) + .4)
            if self.most_resent_msg is None:
                await ctx.send(x, delete_after=t)
            else:
                await self.most_resent_msg.channel.send(x, delete_after=t)

        loop = asyncio.get_running_loop()
        talk = False
        what = ""
        while running:
            que_msg = await loop.run_in_executor(None, self.sender_que.get)
            que_msg = remove_styles(que_msg)

            if que_msg.startswith("#SAY#:"):
                que_msg = que_msg.replace("#SAY#:", "")
                talk = True
                what = que_msg

            if 'exit' == que_msg:
                running = False

            f = len(que_msg) // 2000

            for i in range(f):
                await send(que_msg[:2000])
                que_msg = que_msg[2000:]
            else:
                await send(que_msg)

            if talk:
                talk = False
                aceptedt = "1234567890ßqwertzuiopüasdfghjklöäyxcvbnm,.:?!´éúíóá"
                await eleven_labs_speech_stream(ctx, ''.join(e for e in what if e.lower() in aceptedt), voice_index=self.voice_index)
                what = ""

        await ctx.send("Connection closed")

    async def on_message(self, message):

        if message.author == self.user:
            return
        if not isinstance(message.channel, discord.DMChannel):
            if message.channel.name == 'context':
                self.context.append(message.content)
                self.isaa.get_agent_config_class("self").context.text += message.content
            if message.channel.name == 'augment':
                self.isaa.init_from_augment(eval(message.content))
        if message.content.startswith('v+'):
            if len(voices)-1 > self.voice_index:
                self.voice_index += 1
            await message.reply(f'Confirmed at {self.voice_index}', mention_author=True)
        elif message.content.startswith('v-'):
            if self.voice_index > 0:
                self.voice_index -= 1
            await message.reply(f'Confirmed at {self.voice_index}', mention_author=True)
        elif message.content.startswith('v'):
            try:
                v = int(message.content[1:].strip())
                if len(voices)-1 > v:
                    self.voice_index = v
                    await message.reply(f'Confirmed at {self.voice_index}', mention_author=True)
                else:
                    await message.reply(f'Max value : {len(voices)-1}', mention_author=True)
            except ValueError:
                pass
                await message.reply(f'Invalid int {message.content[1:].strip()}', mention_author=True)

        elif message.content.startswith('exit'):
            self.isaa.on_exit()
            await message.reply('Confirmed', mention_author=True)
            await self.close()

        elif message.content.startswith('list'):
            # Assuming the message.body is storing a list
            msg = ""
            i = 0
            all_chains = list(self.isaa.agent_chain.chains.keys())
            if len(all_chains) == 0:
                msg = "No chains (Skills) found"
            for option in all_chains:
                i += 1
                if len(msg) > 1000:
                    new_mnsg = msg + f" ... total{len(list(self.isaa.agent_chain.chains.keys()))} at {i}\n"
                    print("LEN:", len(new_mnsg), new_mnsg)
                    await message.channel.send(new_mnsg)
                    msg = ""
                msg += f"Name : {option}\nDescription: \n{self.isaa.agent_chain.get_discr(option)}\n\n"
            await message.channel.send(msg)
            await message.channel.send("Done")
        elif message.content.startswith('user-edit'):
            # Collect the next message from the same user
            chain_name_dsd = message.content.split(' ')[-1]
            if chain_name_dsd not in list(self.isaa.agent_chain.chains.keys()):
                await message.channel.send(
                    f'Name {chain_name_dsd} is invalid valid ar {self.isaa.agent_chain.chains.keys()}')
                return
            await message.channel.send(f'```{self.isaa.agent_chain.get(chain_name_dsd)}```')

            def check(m):
                return m.author == message.author and m.channel == message.channel

            try:
                selection = await self.wait_for('message', timeout=60.0 * 15, check=check)
            except asyncio.TimeoutError:
                await message.channel.send('Sorry, time to save your selection is up.')
            else:
                # Saving the selection in a text file
                await message.channel.send(f"New Chain : {selection.content}")
                self.isaa.agent_chain.add(chain_name_dsd, eval(selection.content))

        elif message.content:
            # self.print(self.all_commands)
            self.most_resent_msg = message
            await self.process_commands(message)

    def on_start(self):
        self.logger.info('Starting discordBotManager')
        self.token = os.getenv("DISCORD_BOT_TOKEN")

    def on_exit(self):
        self.stop_bot()
        self.logger.info('Closing discordBotManager')

    def show_version(self):
        self.print('Version: ', self.version)
        return self.version

    def setup_bot(self):
        data = {
            "username": "DiscordBot",
            "email": "DiscordBot@discord.bot",
            "password": "DiscordBot",
            "invitation": "bot-key"
        }
        cloudm: cmTools = self.app.get_mod('cloudM')
        if test_if_exists("DiscordBot", self.app):
            self.app.run_any("DB", "set", ["bot-key", "Valid"])
            self.ws_id = cloudm.create_user([data], self.app)
        else:
            self.ws_id = cloudm.log_in_user([data], self.app)

    def start_bot(self):
        if self.t0 is None:
            self.isaa.global_stream_override = False
            self.sender_que = queue.Queue()
            self.t0 = threading.Thread(target=self.run, args=(self.token,))
            # if not self.ws_id:
            #     self.setup_bot()
            #     time.sleep(6)
            # self.sender_que, self.receiver_que = self.app.run_any("WebSocketManager", "srqw", ['wss://0.0.0.0:5000/ws', self.ws_id])
            self.t0.start()
            return
        self.print("Bot is already running")

    def stop_bot(self):
        if self.t0 is None:
            self.print("No Bot running")
            return
        if self.is_closed():
            self.print("Bot is cosed")
            return

        async def close_bot():
            await self.close()

        loop = asyncio.get_event_loop()
        loop.run_until_complete(close_bot())

        time.sleep(4)
        self.t0.join()
        time.sleep(4)

        del self.t0
        self.t0 = None

    async def run_isaa_with_print(self, ctx, func):
        def run_chain_func():
            try:
                func()
            except Exception as e:
                self.sender_que.put(f'Error details: {e}')
            finally:
                self.sender_que.put('exit')

        t0 = threading.Thread(target=run_chain_func)

        def printer(x, *args, **kwargs):
            self.sender_que.put(str(x))

        _p = self.isaa.print
        self.isaa.print = printer
        t0.start()
        await self.minique_responder(ctx=ctx)
        t0.join()
        self.isaa.print = _p

    def add_commands(self):

        @self.command(name="context", pass_context=True)
        async def context(ctx):
            await ctx.channel.send(str(self.context))
            await ctx.channel.send(f"Description : ```{self.isaa.get_augment()}```")

        @self.command(name="price", pass_context=True)
        async def price(ctx):
            def func():
                self.isaa.show_usage(self.sender_que.put)

            await self.run_isaa_with_print(ctx, func)

        @self.command(name="save2em", pass_context=True)
        async def save2mem(ctx):
            await self.run_isaa_with_print(ctx, self.isaa.save_to_mem)
            await ctx.channel.send("Memory cleared.")
            await ctx.channel.send("How can i help you")

        @self.command(name="ask", pass_context=True)
        async def ask(ctx: commands.Context, *task: str):

            task = ' '.join(task)
            view = Confirm()

            await ctx.send(f"Searching for Matching skills", delete_after=120)

            chain_name, dscription = self.isaa.get_best_fitting(task)

            while '"' in chain_name:
                chain_name = chain_name.replace('"', '')

            if not chain_name in list(self.isaa.agent_chain.chains.keys()):
                await ctx.send(f"crating new chain", delete_after=120)
                chain_name = self.isaa.create_task(task)
                await ctx.send(f'Crated new Chain', delete_after=120)

            run_chain = self.isaa.agent_chain.get(chain_name)
            await ctx.send(f"## Chain details : ```{run_chain}```")
            await ctx.send(f"## Description : ```{self.isaa.agent_chain.get_discr(chain_name)}```")
            await ctx.send(f"## Why : ```{dscription}```")
            betwean_res = self.isaa.mini_task_completion(
                f"Genearate an Output for a aget to be spoken loud. Informations {chain_name} {dscription} tell the user why you want to run this chin :")
            await eleven_labs_speech_stream(ctx, betwean_res, voice_index=self.voice_index)
            await ctx.send(f'Do you want to continue? with {chain_name}', view=view)
            # Wait for the View to stop listening for input...
            await view.wait()
            if view.value is None:
                print('Timed out...')
            elif view.value:
                print('Confirmed...')
                await ctx.send(f"running chain ... pleas wait")

                def func():
                    res = self.isaa.execute_thought_chain(task, run_chain, self.isaa.get_agent_config_class("self"))
                    self.sender_que.put(f"#SAY#:{res[0]}")
                    if len(res) == 2:
                        self.sender_que.put(f"Proses Summary : ```{res[0]}```")
                        if isinstance(len(res[-1]), str):
                            self.sender_que.put(f"response : ```{res[-1]}```")
                        if isinstance(len(res[-1]), list):
                            if isinstance(len(res[-1][-1]), str):
                                self.sender_que.put(f"response : ```{res[-1][-1]}```")

                    # self.isaa.print = print_sto
                    self.sender_que.put(f"returned : ```{self.app.pretty_print(list(res))}```")

                await self.run_isaa_with_print(ctx, func)
            else:
                print('Cancelled...')

        @self.command(name="create", pass_context=True)
        async def create(ctx: commands.Context, *text: str):
            # await ctx.send(f"Online Type your massage ... (start witch isaa)")
            # def check(msg):
            #     return msg.author == ctx.author and msg.channel == ctx.channel and \
            #         msg.content.startswith("isaa")
            # msg = await self.wait_for("message", check=check, timeout=60 * 30)
            task = ' '.join(text)
            name = self.isaa.create_task(task)
            task_chain = self.isaa.agent_chain.get(name)

            msg = f"""# New Task Crated
             ## Name : {name}
             ### task structure ```{task_chain}```"""
            await ctx.send(msg)

            def run_chain_func():
                try:
                    dis = self.isaa.describe_chain(name)
                    self.sender_que.put(f"return {dis}")
                except Exception as e:
                    self.sender_que.put(f'Error details: {e}')
                finally:
                    self.sender_que.put('exit')

            t0 = threading.Thread(target=run_chain_func)

            def printer(x, *args, **kwargs):
                self.sender_que.put(str(x))

            _p = self.isaa.print
            self.isaa.print = printer
            t0.start()
            await self.minique_responder(ctx=ctx)
            t0.join()
            self.isaa.print = _p
            await ctx.send("Done")

        @self.command(name="google", pass_context=True)
        async def google(ctx: commands.Context, *task: str):
            task = ' '.join(task)

            await ctx.send("We found :", view=Google(task))

        @self.command(name="run", pass_context=True)
        async def run(ctx: commands.Context, *task: str):

            task = ' '.join(task)
            all_chains = list(self.isaa.agent_chain.chains.keys())
            chain_name = all_chains[0]  # self.isaa.get_best_fitting(task)
            task_chain = self.isaa.agent_chain.get(chain_name)

            msg = f"""# Task
             ## ```{task}```
             ## get_best_fitting {chain_name}
             ## details
             ```{task_chain}```"""

            async def run_by_name(chain_name_):
                run_chain = self.isaa.agent_chain.get(chain_name_)
                self.print(f"Chin len {chain_name_}:{len(run_chain)}")
                await ctx.send(f"Chin len : {len(run_chain)}")
                res = "No chain to run"
                if run_chain:
                    await ctx.send(f"chain : ```{run_chain}```")
                    await ctx.send(f"Description : ```{self.isaa.agent_chain.get_discr(chain_name_)}```")
                    await ctx.send(f"running chain ... pleas wait")

                    def func():
                        res = self.isaa.execute_thought_chain(task, run_chain,
                                                              self.isaa.get_agent_config_class("self"))
                        self.sender_que.put(f"#SAY#:{res[0]}")
                        self.sender_que.put(f"return : ```{res}```")

                    await self.run_isaa_with_print(ctx, func)

            await ctx.send(msg, view=TaskSelectionView(run_by_name, all_chains, chain_name, self.isaa.agent_chain))

        @self.command(name="deleat", pass_context=True)
        async def deleat(ctx: commands.Context):

            all_chains = list(self.isaa.agent_chain.chains.keys())
            chain_name = all_chains[0]
            msg = f"""# Tasks: """

            async def run_by_name(chain_name_):
                run_chain = self.isaa.agent_chain.get(chain_name_)
                self.print(f"Chin len {chain_name_}:{len(run_chain)}")
                await ctx.send(f"Chin len : {len(run_chain)}")
                if run_chain:
                    self.isaa.agent_chain.remove(chain_name_)
                await ctx.send(f"Don")

            await ctx.send(msg, view=TaskSelectionView(run_by_name, all_chains, chain_name, self.isaa.agent_chain))

        @self.command(name="describe", pass_context=True)
        async def describe(ctx: commands.Context):

            all_chains = list(self.isaa.agent_chain.chains.keys())
            chain_name = all_chains[0]
            msg = f"""# Tasks: """

            async def run_by_name(chain_name_):
                def func():
                    run_chain = self.isaa.agent_chain.get(chain_name_)
                    self.print(f"Chin len {chain_name_}:{len(run_chain)}")
                    self.sender_que.put(f"Chin len : {len(run_chain)}")
                    if run_chain:
                        self.isaa.describe_chain(chain_name_)

                await self.run_isaa_with_print(ctx, func)
                await ctx.send(f"Don")

            await ctx.send(msg, view=TaskSelectionView(run_by_name, all_chains, chain_name, self.isaa.agent_chain))

        @self.command(name="optimise", pass_context=True)
        async def optimise(ctx: commands.Context):

            all_chains = list(self.isaa.agent_chain.chains.keys())

            def send_to_think(x, *args, **kwargs):
                async def do_send(xy):
                    channel = discord.utils.get(ctx.guild.channels, name="system")
                    if channel is None:
                        channel = await ctx.guild.create_text_channel("system")
                    await channel.send(xy)

                # loop = asyncio.get_event_loop()
                # loop.run_until_complete(do_send(remove_styles(x)))
                # loop.run_until_complete(do_send(x))
                # if args:
                #    loop.run_until_complete(do_send(str(args)))
                # if kwargs:
                #    loop.run_until_complete(do_send(str(kwargs)))

            # print_sto = self.isaa.print
            # self.isaa.print = send_to_think

            msg = f"""What task do you want to optimise"""

            async def run_by_name(chain_name_):
                run_chain = self.isaa.agent_chain.get(chain_name_)
                self.print(f"Chin len {chain_name_}:{len(run_chain)}")
                await ctx.send(f"Chin len : {len(run_chain)}")

                if run_chain:
                    await ctx.send(f"return : ```{run_chain}```")
                    await ctx.send(f"Description : ```{self.isaa.agent_chain.get_discr(chain_name_)}```")
                    await ctx.send(f"optimise chain ... pleas wait")

                    def func():

                        new_task_dict = self.isaa.optimise_task(chain_name_)
                        if not new_task_dict:
                            self.sender_que.put(f"Optimisation Failed")
                            # self.isaa.print = print_sto
                            return
                        self.isaa.agent_chain.add_task(chain_name_ + "-Optimised", new_task_dict)
                        self.sender_que.put(
                            f"return : ```{self.app.pretty_print(list(new_task_dict))} {new_task_dict}```")

                    await self.run_isaa_with_print(ctx, func)

                    # self.isaa.print = print_sto
                await ctx.send(f"Done")

            if len(all_chains) > 25:
                all_chains = all_chains[:25]
            await ctx.send(msg,
                           view=TaskSelectionView(run_by_name, all_chains, all_chains[0], self.isaa.agent_chain))

        @self.command(name="say", pass_context=True)
        async def say(ctx: commands.Context, *text: str):

            await ctx.send("Running")

            await eleven_labs_speech_stream(ctx, ' '.join(text), voice_index=self.voice_index)

            await ctx.send("Ok")

        @self.command(name="chat", pass_context=True)
        async def chat(ctx: commands.Context, *text: str):

            await ctx.send("Running")
            text = ' '.join(text)

            def func():
                res = self.isaa.stream_read_llm(text,
                                                self.isaa.get_agent_config_class("self").
                                                set_mode("free").
                                                set_completion_mode("chat"))
                if not isinstance(res, str):
                    res = str(res)
                self.sender_que.put("# Isaa: " + res)

            await self.run_isaa_with_print(ctx, func)

            await ctx.send("Ok")

        @self.command(name="talk", pass_context=True)
        async def talk(ctx: commands.Context, *text: str):

            await ctx.send("Running")
            text = ' '.join(text)

            def func():
                res = self.isaa.stream_read_llm(text,
                                                self.isaa.get_agent_config_class("self").
                                                set_mode("free").
                                                set_completion_mode("chat"))
                if not isinstance(res, str):
                    res = str(res)
                self.sender_que.put("#SAY#:" + res)

            await self.run_isaa_with_print(ctx, func)

            await ctx.send("Ok")

        @self.command(name="say-in", pass_context=True)
        async def say(ctx: commands.Context, lang: str, *text: str):

            await ctx.send("Running")

            translation = self.isaa.mini_task_completion(f"Translate : '''{text}'''' to :{lang}\nTranslation:")
            await eleven_labs_speech_stream(ctx, translation, voice_index=self.voice_index)

            await ctx.send("Ok")

        @self.command(name='join', pass_context=True)
        async def join(ctx):
            if ctx.author.voice is None:
                await ctx.send("Du bist nicht in einem Sprachkanal!")
                return
            voice_channel = ctx.author.voice.channel
            if ctx.voice_client is None:
                await voice_channel.connect()
            else:
                await ctx.voice_client.move_to(voice_channel)

        @self.command(name='leave', pass_context=True)
        async def leave(ctx):
            if ctx.voice_client is not None:
                await ctx.voice_client.disconnect()
            else:
                await ctx.send("Ich bin nicht in einem Sprachkanal.")

        @self.command(name='welcome', pass_context=True)
        async def welcome(ctx):
            msg = """# 🎉 Welcome to our powerful Discord bot! 🎉

This bot is designed to help you with a variety of tasks and enhance your experience on Discord. It can listen (not voice in chat), respond to commands, interact with AI assistants like Isaa, and provide many other unique and interesting features. ✨

Please follow the commands listed below to get the best results:"""
            await ctx.send(msg)
            msg = """🔹 join - The bot joins the voice channel.
🔹 leave - Leaves the voice channel.
🔹 ask - Asks Isaa a question. If no matching skill is found, one is created.
🔹 chat - Chat with a smaller model.
🔹 context - Displays the context and status of the Isaa system.
🔹 create - Creates a task.
🔹 delete - Deletes a task.
🔹 describe - Describes a task.
🔹 google - Performs a Google search.
🔹 help - Displays this message.
🔹 optimize - Optimizes a task.
🔹 price - Displays the current usage.
🔹 run - Executes a task.
🔹 save2em - Clears the memory.
🔹 say - Speaks in the voice channel the user is in. If the bot is not in the channel, it joins it.
🔹 say-in - Similar to say, but the first argument is the language to speak in.
🔹 talk - Makes the agent talk in the background with 'say'.
🔹 talkr - Similar to talk, but the agent can use tools.
🔹 v+ - Selects the next voice.
🔹 v- - Selects the previous voice.
🔹 list - Lists all tasks.
🔹 user-edit $Task-name - Processes a task. The first argument is the task name.
🔹 exit - Terminates the agent.
🔹 welcome - This msg.
🔹 file - add a file to context in 'content' agent """
            await ctx.send(msg)
            msg = """Have fun trying out these commands and exploring the many features our bot has to offer! If you
have any questions, don't hesitate to contact us. We hope to provide you with an efficient and enjoyable
experience on Discord with our Discord bot. 🌟"""
            await ctx.send(msg)

        @self.command(name='file', pass_context=True)
        async def read_file(ctx):
            if not ctx.message.attachments:
                await ctx.send("You need to send a text file with this command.")
                return
            attachment_url = ctx.message.attachments[0].url
            async with aiohttp.ClientSession() as session:
                async with session.get(attachment_url) as resp:
                    if resp.status != 200:
                        await ctx.send('Could not download file...')
                        return
                    data = await resp.text()
            # Now you can process your 'data'
            await ctx.send(f"data len is {len(data)}")

            res = self.isaa.mas_text_summaries(data)
            self.isaa.get_agent_config_class("content").context.text += res

    async def on_member_join(self, member):

        await member.send('Welcome to the server! pleas writ on the new chat the i crated on the server for u')
        channel = await member.guild.create_text_channel(str(member.name).replace(' ', '-'))
        await channel.send(f"Welcome {str(member.name).replace(' ', '-')} to your new channel!")
        msg = """# 🎉 Welcome to our powerful Discord bot! 🎉

 This bot is designed to help you with a variety of tasks and enhance your experience on Discord. It can listen (not voice in chat), respond to commands, interact with AI assistants like Isaa, and provide many other unique and interesting features. ✨

Please follow the commands listed below to get the best results:"""
        await channel.send(msg)
        msg = """🔹 join - The bot joins the voice channel.
🔹 leave - Leaves the voice channel.
🔹 ask - Asks Isaa a question. If no matching skill is found, one is created.
🔹 chat - Chat with a smaller model.
🔹 context - Displays the context and status of the Isaa system.
🔹 create - Creates a task.
🔹 delete - Deletes a task.
🔹 describe - Describes a task.
🔹 google - Performs a Google search.
🔹 help - Displays this message.
🔹 optimize - Optimizes a task.
🔹 price - Displays the current usage.
🔹 run - Executes a task.
🔹 save2em - Clears the memory.
🔹 say - Speaks in the voice channel the user is in. If the bot is not in the channel, it joins it.
🔹 say-in - Similar to say, but the first argument is the language to speak in.
🔹 talk - Makes the agent talk in the background with 'say'.
🔹 talkr - Similar to talk, but the agent can use tools.
🔹 v+ - Selects the next voice.
🔹 v- - Selects the previous voice.
🔹 list - Lists all tasks.
🔹 user-edit $Task-name - Processes a task. The first argument is the task name.
🔹 exit - Terminates the agent.
🔹 welcome - This msg.
🔹 file - add a file to context in 'content' agent """
        await channel.send(msg)
        msg = """Have fun trying out these commands and exploring the many features our bot has to offer! If you
have any questions, don't hesitate to contact us. We hope to provide you with an efficient and enjoyable
experience on Discord with our Discord bot. 🌟"""
        await channel.send(msg)


voices = ["27dJPQc4TXmS1pccxP0m", "VzRk86yeIgj45NCVZqJe", "e3sRAASduwyXKQwXY3ci", "onwK4e9ZLuTAKqWW03F9",
          "CYw3kZ02Hs0563khs1Fj", "ThT5KcBeYPX3keUQqHPh", "MF3mGyEYCl7XYWbV9V6O", "LcfcDJNUP1GQjkzn1xUU",
          "jsCqWAovK2LkecY7zXl4", "zcAOhNBS3c14rBihAFp1", "EXAVITQu4vr4xnSDxMaL", "9Mi9dBkaxn2pCIdAAGOB"]


async def play_audio_stream(ctx, audio_stream):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        chunk_size = 1024  # You can adjust this value depending on your needs
        while True:
            chunk = audio_stream.read(chunk_size)
            if not chunk:
                break
            temp_file.write(chunk)
        temp_file.flush()
        if not ctx.message.author.voice:
            await ctx.send("You are not connected to a voice channel")
            return
        voice_channel = ctx.message.author.voice.channel
        if ctx.voice_client is not None:  # bot is already connected to a voice channel
            if ctx.voice_client.channel == voice_channel:  # bot is in the same channel as the author
                vc = ctx.voice_client
            else:  # bot is in a different channel, so move to author's channel
                await ctx.voice_client.move_to(voice_channel)
                vc = ctx.voice_client
        else:  # bot is not connected to any voice channel, so connect to author's channel
            vc = await voice_channel.connect()
        vc.play(discord.FFmpegPCMAudio(executable="ffmpeg", source=temp_file.name))
        while vc.is_playing():
            await asyncio.sleep(1)
        # await vc.disconnect()
        await ctx.send("Completed", delete_after=15)


# GBv7mTt0atIp3Br8iCZE/98542988-5267-4148-9a9e-baa8c4f14644.mp3
# GBv7mTt0atIp3Br8iCZE

# 4VrpCbKeaHwMPrbLPOH

async def eleven_labs_speech_stream(ctx, text, voice_index=0, voices_=None, print=print):
    if not ctx.message.author.voice:
        await ctx.send("You are not connected to a voice channel")
        return
    if voices_ is None:
        voices_ = voices
    tts_headers = {
        "Content-Type": "application/json",
        "xi-api-key": os.getenv("ELEVENLABS_API_KEY")
    }
    tts_url = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream".format(
        voice_id=voices_[voice_index])
    formatted_message = {"text": text}
    response = requests.post(
        tts_url, headers=tts_headers, json=formatted_message, stream=True)
    if response.status_code == 200:
        await play_audio_stream(ctx, response.raw)
        return True
    else:
        print("Request failed with status code:", response.status_code)
        print("Response content:", response.content)
        return False


