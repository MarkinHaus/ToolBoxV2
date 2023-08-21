import asyncio
import os
import threading
import time
from urllib.parse import quote_plus

import discord
from discord.ext import commands
from toolboxv2 import MainTool, FileHandler, App, Style, remove_styles
from toolboxv2.mods.isaa import Tools as isaaTools


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
    def __init__(self, run_chain_callback, options, selected, chains, ctx):
        super().__init__(timeout=60 * 15)
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
            options_select.append(discord.SelectOption(value=_, label=_, description=dis))

        async def callback_fuc(val, send):
            if val == "1":
                await send("Running chain ...")
                await run_chain_callback(selected)
            elif val == "-1":
                return "Abort"
            else:
                if val in options:
                    await send(f"Running chain {val}...")
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
        MainTool.__init__(self, load=self.on_start, v=self.version, tool=self.tools, name=self.name, logs=self.logger,
                          color=self.color, on_exit=self.on_exit)
        self.isaa: isaaTools = app.get_mod('isaa')

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

    async def on_message(self, message):
        if message.author == self.user:
            return
        if message.channel.name == 'context':
            self.context.append(message.content)
        elif message.content.startswith('hello'):
            await message.reply('Hello!', mention_author=True)
        elif message.content.startswith('list'):
            # Assuming the message.body is storing a list
            msg = ""
            i = 0
            for option in list(self.isaa.agent_chain.chains.keys()):
                i += 1
                if len(msg) > 1000:
                    new_mnsg = msg+f" ... total{len(list(self.isaa.agent_chain.chains.keys()))} at {i}\n"
                    print("LEN:", len(new_mnsg), new_mnsg)
                    await message.channel.send(new_mnsg)
                    msg = ""
                msg += f"Name : {option}\n\nnDescription \n:{self.isaa.agent_chain.get_discr(option)}\n\n"
            await message.channel.send(msg)
            await message.channel.send("Done")
        elif message.content.startswith('user-edit'):
            # Collect the next message from the same user
            chain_name_dsd = message.content.split(' ')[-1]
            if chain_name_dsd in list(self.isaa.agent_chain.chains.keys()):
                await message.channel.send(f'```{self.isaa.agent_chain.get(chain_name_dsd)}```')

            def check(m):
                return m.author == message.author and m.channel == message.channel

            try:
                selection = await self.wait_for('message', timeout=60.0*15, check=check)
            except asyncio.TimeoutError:
                await message.channel.send('Sorry, time to save your selection is up.')
            else:
                # Saving the selection in a text file
                await message.channel.send(f"New Chain : {selection.content}")
                self.isaa.agent_chain.add(chain_name_dsd, eval(selection.content))

        elif message.content:
            self.print(self.all_commands)
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

    def start_bot(self):
        if self.t0 is None:
            self.t0 = threading.Thread(target=self.run, args=(self.token,))
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

    def add_commands(self):

        @self.command(name="context", pass_context=True)
        async def context(ctx):
            await ctx.channel.send(str(self.context))

        @self.command(name="price", pass_context=True)
        async def price(ctx):

            def send_to_think(x, *args, **kwargs):

                async def do_send(xy):
                    await ctx.send(xy)

                loop = asyncio.get_event_loop()
                loop.run_until_complete(do_send(remove_styles(x)))

            await ctx.channel.send(str(self.isaa.show_usage(send_to_think)))

        @self.command(name="ask", pass_context=True)
        async def ask(ctx: commands.Context):
            await ctx.send(f"Online Type your massage ... (start witch isaa)")

            def check(msg):
                return msg.author == ctx.author and msg.channel == ctx.channel and \
                    msg.content.startswith("isaa")

            msg = await self.wait_for("message", check=check, timeout=60 * 30)
            task = msg.content[4:]
            task = ' '.join(task)
            """Asks the user a question to confirm something."""
            # We create the view and assign it to a variable so we can wait for it later.
            view = Confirm()

            def send_to_think(x, *args, **kwargs):
                async def do_send(xy):
                    channel = discord.utils.get(ctx.guild.channels, name="system")
                    if channel is None:
                        channel = await ctx.guild.create_text_channel("system")
                    await channel.send(xy)

                #loop = asyncio.get_event_loop()
                #loop.run_until_complete(do_send(remove_styles(x)))
                #loop.run_until_complete(do_send(x))
                #if args:
                #    loop.run_until_complete(do_send(str(args)))
                #if kwargs:
                #    loop.run_until_complete(do_send(str(kwargs)))

            #print_sto = self.isaa.print
            #self.isaa.print = send_to_think

            chain_name = self.isaa.get_best_fitting(task)

            while '"' in chain_name:
                chain_name = chain_name.replace('"', '')

            if not chain_name in list(self.isaa.agent_chain.chains.keys()):
                chain_name = self.isaa.create_task(task)
                await ctx.send(f'Crated new Chain')

            run_chain = self.isaa.agent_chain.get(chain_name)
            await ctx.send(f"Chain details : ```{run_chain}```")
            await ctx.send(f"Description : ```{self.isaa.agent_chain.get_discr(chain_name)}```")
            await ctx.send(f'Do you want to continue? with {chain_name}', view=view)
            # Wait for the View to stop listening for input...
            await view.wait()
            if view.value is None:
                print('Timed out...')
            elif view.value:
                print('Confirmed...')
                await ctx.send(f"running chain ... pleas wait")
                res = self.isaa.execute_thought_chain(task, run_chain, self.isaa.get_agent_config_class("self"))
                if len(res) == 2:
                    await ctx.send(f"Proses Summary : ```{self.app.pretty_print(list(res[0]))}```")
                    if isinstance(len(res[-1]), str):
                        await ctx.send(f"response : ```{res[-1]}```")
                    if isinstance(len(res[-1]), list):
                        if isinstance(len(res[-1][-1]), str):
                            await ctx.send(f"response : ```{res[-1][-1]}```")

                #self.isaa.print = print_sto
                await ctx.send(f"returned : ```{self.app.pretty_print(list(res))}```")
            else:
                print('Cancelled...')

        @self.command(name="create", pass_context=True)
        async def create(ctx: commands.Context):
            await ctx.send(f"Online Type your massage ... (start witch isaa)")

            def check(msg):
                return msg.author == ctx.author and msg.channel == ctx.channel and \
                    msg.content.startswith("isaa")

            msg = await self.wait_for("message", check=check, timeout=60 * 30)
            task = msg.content[4:]
            name = self.isaa.create_task(task)
            task_chain = self.isaa.agent_chain.get(name)

            msg = f"""# New Task Crated
             ## Name : {name}
             ### task structure ```{task_chain}```"""
            await ctx.send(msg)

            dis = self.isaa.describe_chain(name)

            await ctx.send(dis)

        @self.command(name="google", pass_context=True)
        async def google(ctx: commands.Context, *task: str):
            task = ' '.join(task)

            await ctx.send("We found :", view=Google(task))

        @self.command(name="run", pass_context=True)
        async def run(ctx: commands.Context, *task: str):
            await ctx.send(f"Online Type your massage ... (start witch isaa)")

            def check(msg):
                return msg.author == ctx.author and msg.channel == ctx.channel and \
                    msg.content.startswith("isaa")

            msg = await self.wait_for("message", check=check, timeout=60*30)
            task = msg.content[4:]
            all_chains = list(self.isaa.agent_chain.chains.keys())
            chain_name = all_chains[0]  # self.isaa.get_best_fitting(task)
            task_chain = self.isaa.agent_chain.get(chain_name)

            def send_to_think(x, *args, **kwargs):
                async def do_send(xy):
                    channel = discord.utils.get(ctx.guild.channels, name="system")
                    if channel is None:
                        channel = await ctx.guild.create_text_channel("system")
                    await channel.send(xy)

                #loop = asyncio.get_event_loop()
                #loop.run_until_complete(do_send(remove_styles(x)))
                #loop.run_until_complete(do_send(x))
                #if args:
                #    loop.run_until_complete(do_send(str(args)))
                #if kwargs:
                #    loop.run_until_complete(do_send(str(kwargs)))

            #print_sto = self.isaa.print
            #self.isaa.print = send_to_think

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
                    await ctx.send(f"return : ```{run_chain}```")
                    await ctx.send(f"Description : ```{self.isaa.agent_chain.get_discr(chain_name_)}```")
                    await ctx.send(f"running chain ... pleas wait")
                    res = self.isaa.execute_thought_chain(task, run_chain, self.isaa.get_agent_config_class("self"))
                    #self.isaa.print = print_sto
                await ctx.send(f"return : ```{self.app.pretty_print(list(res))} {res}```")

            await ctx.send(msg, view=TaskSelectionView(run_by_name, all_chains, chain_name, self.isaa.agent_chain, ctx))

        @self.command(name="edit", pass_context=True)
        async def edit(ctx: commands.Context):

            all_chains = list(self.isaa.agent_chain.chains.keys())

            def send_to_think(x, *args, **kwargs):
                async def do_send(xy):
                    channel = discord.utils.get(ctx.guild.channels, name="system")
                    if channel is None:
                        channel = await ctx.guild.create_text_channel("system")
                    await channel.send(xy)

                #loop = asyncio.get_event_loop()
                #loop.run_until_complete(do_send(remove_styles(x)))
                #loop.run_until_complete(do_send(x))
                #if args:
                #    loop.run_until_complete(do_send(str(args)))
                #if kwargs:
                #    loop.run_until_complete(do_send(str(kwargs)))

            #print_sto = self.isaa.print
            #self.isaa.print = send_to_think

            msg = f"""What task do you want to optimise"""

            async def run_by_name(chain_name_):
                run_chain = self.isaa.agent_chain.get(chain_name_)
                self.print(f"Chin len {chain_name_}:{len(run_chain)}")
                await ctx.send(f"Chin len : {len(run_chain)}")
                new_task_dict = ["No chain to edit"]
                if run_chain:
                    await ctx.send(f"return : ```{run_chain}```")
                    await ctx.send(f"Description : ```{self.isaa.agent_chain.get_discr(chain_name_)}```")
                    await ctx.send(f"running chain ... pleas wait")
                    new_task_dict = self.isaa.optimise_task(chain_name_)
                    if not new_task_dict:
                        await ctx.send(f"Optimisation Failed")
                        #self.isaa.print = print_sto
                        return
                    self.isaa.agent_chain.add_task(chain_name_+"-Optimised", new_task_dict)
                    #self.isaa.print = print_sto
                await ctx.send(f"return : ```{self.app.pretty_print(list(new_task_dict))} {new_task_dict}```")

            await ctx.send(msg, view=TaskSelectionView(run_by_name, all_chains, all_chains[0], self.isaa.agent_chain, ctx))
